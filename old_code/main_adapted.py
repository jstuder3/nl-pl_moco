"""
Adapted from: https://github.com/PyTorchLightning/lightning-bolts/blob/5ab4faeaa4eca378b24d22e18316a2be4e9745b3/pl_bolts/models/self_supervised/moco/moco2_module.py#L339

### WILL NEED TO ADD LICENSING STUFF HERE BECAUSE I'M PRETTY SURE I CAN'T JUST ADAPT IT WITHOUT SOMETHING LIKE THAT
"""
from argparse import ArgumentParser
from typing import Union

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModel

from pl_bolts.metrics import mean, precision_at_k

from utils.data_loading import generateDataLoader

import platform

from datetime import date

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#if platform.system() == "Linux":
   # from pl_bolts.utils import _PL_GREATER_EQUAL_1_4  # THIS WILL CRASH ON WINDOWS! MAKE SURE TO NOT USE DDP ON WINDOWS!

class Moco_v2(LightningModule):

    def __init__(
        self,
        base_encoder: Union[str, torch.nn.Module] = "microsoft/codebert-base",
        emb_dim: int = 128,
        num_negatives: int = 32768,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 1e-5,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        data_dir: str = './',
        batch_size: int = 16,
        num_workers: int = 8,
        augment: bool = True,
        *args,
        **kwargs
    ):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            data_dir: the directory to store data
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder, self.momentum_encoder = self.init_encoders(base_encoder)

        self.encoder_mlp = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 128))
        self.momentum_encoder_mlp = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 128))

        for param_k in self.encoder.parameters():
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the validation queue
        self.register_buffer("val_queue", torch.randn(emb_dim, num_negatives))
        self.val_queue = nn.functional.normalize(self.val_queue, dim=0)

        self.register_buffer("val_queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self, base_encoder):
        """
        Override to add your own encoders
        """
        encoder_q = AutoModel.from_pretrained(base_encoder)
        encoder_k = AutoModel.from_pretrained(base_encoder)

        return encoder_q, encoder_k

    def train_dataloader(self): #generates a freshly augmented dataset
        return  generateDataLoader("code_search_net", "python", f"train[:{args.train_split_size}%]", tokenizer, batch_size=batch_size, shuffle=True, augment=True)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ptr, queue):
        # gather keys before updating queue
        if platform.system()=="Linux":
            if self._use_ddp_or_ddp2(self.trainer):
                keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        queue_ptr[0] = ptr

    def forward(self, docs_batch, code_batch, queue, isInference=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            queue: a queue from which to pick negative samples
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder(input_ids=docs_batch["input_ids"], attention_mask=docs_batch["attention_mask"])["pooler_output"]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        q = self.encoder_mlp(q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            k = self.encoder(input_ids=code_batch["input_ids"], attention_mask=code_batch["attention_mask"])["pooler_output"]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            if isInference:
                k = self.momentum_encoder_mlp(k)
                k = nn.functional.normalize(k, dim=1)

        if not isInference: #want to compute gradients of MLP if not doing inference
            k = self.momentum_encoder_mlp(k)
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        return logits, labels, k

    def training_step(self, batch, batch_idx):

        docs = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}

        self._momentum_update_key_encoder()  # update the key encoder
        output, target, keys = self.forward(docs_batch=docs, code_batch=code, queue=self.queue)
        self._dequeue_and_enqueue(keys, queue=self.queue, queue_ptr=self.queue_ptr)  # dequeue and enqueue

        loss = F.cross_entropy(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {'train_loss': loss, 'train_acc1': acc1, 'train_acc5': acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):

        docs = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}
        #current_batch_size=docs["input_ids"].shape[0]
        #labels = torch.tensor([0 for h in range(current_batch_size)]).cuda()

        output, target, keys = self.forward(docs_batch=docs, code_batch=code, queue=self.val_queue, isInference=True)
        self._dequeue_and_enqueue(keys, queue=self.val_queue, queue_ptr=self.val_queue_ptr)  # dequeue and enqueue

        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {'val_loss': val_loss, 'val_acc1': val_acc1, 'val_acc5': val_acc5}
        self.log_dict(log)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--base_encoder', type=str, default="microsoft/codebert-base")
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--queue_size', type=int, default=32768) # might not work
        parser.add_argument("--num_epochs", type=int, default=10)
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=16) #this might also not work for epochs after the first one
        parser.add_argument("--train_split_size", type=int, default=1)
        parser.add_argument("--val_split_size", type=int, default=5)
        parser.add_argument("--augment", type=bool, default=True) # this does not work! will need to fix this later

        return parser

    @staticmethod
    def _use_ddp_or_ddp2(trainer: Trainer) -> bool:
        # for backwards compatibility
        #if _PL_GREATER_EQUAL_1_4:
        #    return trainer.accelerator_connector.use_ddp or trainer.accelerator_connector.use_ddp2
        #return trainer.use_ddp or trainer.use_ddp2
        return platform.system()=="Linux"

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def execute():

    parser = ArgumentParser()

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = Moco_v2.add_model_specific_args(parser)
    args = parser.parse_args()

    batch_size = args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(args.base_encoder)
    val_loader = generateDataLoader("code_search_net", "python", f"validation[:{args.val_split_size}%]", tokenizer, batch_size=batch_size, shuffle=False, augment=False)

    model = Moco_v2(**args.__dict__)

    batch_size=args.batch_size
    date_formatted = date.today().strftime("%b-%d-%Y-%H-%M")
    logger = pl.loggers.TensorBoardLogger("./runs/", name=f"{date_formatted}-batch_size={batch_size}-queue_size={args.queue_size}-max_epochs={args.num_epochs}-train_split={args.train_split_size}-val_split={args.val_split_size}-num_gpus={torch.cuda.device_count()}")

    # IMPORTANT: FOR TESTING ON WINDOWS, USE EITHER DP OR DDP_CPU BECAUSE DDP IS NOT SUPPORTED
    trainer = pl.Trainer(gpus=-1, max_epochs=args.num_epochs, logger=logger, precision=16, accelerator=("ddp" if platform.system()=="Linux" else "dp"), reload_dataloaders_every_n_epochs=1, plugins="deepspeed")
    # remove log_gpu_memory and fast_dev_run later because it may slow down training

    #for _ in range(args.num_epochs):
        # generate new augmented dataset
    train_loader = generateDataLoader("code_search_net", "python", f"train[:{args.train_split_size}%]", tokenizer, batch_size=batch_size, shuffle=True, augment=True)
        # train for one epoch
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    execute()

# LOTS OF DEBUGGING STUFF BECAUSE I WANT TO GET DEEPSPEED TO WORK FIRST! THIS IS BY NO MEANS WHAT xMoCo SHOULD LOOK LIKE!

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
import argparse
import random
from datetime import datetime
import numpy as np

from utils.multimodal_data_loading import generateDataLoader
from utils.metric_computation import validation_computations

class xMoCoModelPTL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.docs_tokenizer = None
        self.code_tokenizer = None

        self.docs_fast_encoder = AutoModel.from_pretrained(args.docs_encoder)
        #self.docs_slow_encoder = AutoModel.from_pretrained(args.docs_encoder)
        #self.code_fast_encoder = AutoModel.from_pretrained(args.code_encoder)
        #self.code_slow_encoder = AutoModel.from_pretrained(args.code_encoder)

        self.effective_queue_size = self.args.effective_queue_size
        self.update_weight = self.args.momentum_update_weight
        self.batch_size = self.args.effective_batch_size
        self.num_gpus = args.num_gpus

        # debug: remove later
        self.register_buffer("queue", torch.randn(int(self.batch_size/self.num_gpus), 768))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def load_tokenizers(self):
        if self.docs_tokenizer == None:
            self.docs_tokenizer = AutoTokenizer.from_pretrained(self.args.docs_encoder)
        if self.code_tokenizer == None:
            self.code_tokenizer = AutoTokenizer.from_pretrained(self.args.code_encoder)

    @torch.no_grad()
    def train_dataloader(self):
        self.load_tokenizers()
        train_loader = generateDataLoader("python", "train", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=self.args.shuffle, augment=self.args.augment, num_workers=self.args.num_workers)#int(math.floor(multiprocessing.cpu_count()/torch.cuda.device_count())))
        return train_loader

    @torch.no_grad()
    def val_dataloader(self):
        self.load_tokenizers()
        val_loader = generateDataLoader("python", "valid", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=False, augment=False, num_workers=self.args.num_workers)  # int(math.floor(multiprocessing.cpu_count()/torch.cuda.device_count())))
        return val_loader

    # ADAPTED FROM https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/moco/moco2_module.py#L137
    @torch.no_grad()
    def updateMomentumEncoder(self, fast_encoder, slow_encoder):
        """Momentum update of the key encoder."""
        for param_fast, param_slow in zip(self.fast_encoder.parameters(), self.slow_encoder.parameters()):
            em = self.update_weight
            param_slow.data = param_slow.data * em + param_fast.data * (1.0 - em)

    def forward(self, docs_samples, code_samples):
        # debug: change later
        out1=self.docs_fast_encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
        with torch.no_grad():
            out2=self.docs_fast_encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
        return out1, out2

    def training_step(self, batch, batch_idx):
        # debug: change later
        docs_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}
        batch_indices = batch["index"]

        current_batch_size = docs_samples["input_ids"].shape[0]

        docs_embeddings, code_embeddings = self.forward(docs_samples, code_samples) # after deepspeed tests: make this return everything, including the slow_encoder embeddings

        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        l_pos = torch.bmm(docs_embeddings.view((current_batch_size, 1, -1)), code_embeddings.view((current_batch_size, -1, 1)))

        if code_embeddings.shape[0] != 0:
            # compute similarity of negative NL/PL pairs and concatenate with l_pos to get logits
            l_neg = torch.matmul(docs_embeddings.view((current_batch_size, -1)), torch.transpose(self.queue, 0, 1))
            logits = torch.cat((l_pos.view((current_batch_size, 1)), l_neg), dim=1)

        labels = torch.tensor([0 for h in range(code_embeddings.shape[0])]).type_as(code_samples["input_ids"])

        loss = nn.CrossEntropyLoss()(logits / self.args.temperature, labels)

        self.queue[:] = code_embeddings.detach()

        return loss

    def validation_step(self, batch, batch_idx):
        pass

# COPIED FROM https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/codesearch/run_classifier.py#L45
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed) # I don't use numpy, but some library I use here might use numpy as a backend
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def execute(args):
    # set all seeds so we can ensure same MLP initialization and augmentation behaviour on all GPUS
    set_seed(args.seed)

    model = xMoCoModelPTL(args)

    now = datetime.now()
    now_str = now.strftime("%b%d_%H_%M_%S")
    logger = pl.loggers.TensorBoardLogger("runs", name=f"{now_str}-effective_bs_{args.effective_batch_size}-lr_{args.learning_rate}-effective_queue_size_{args.effective_queue_size}-max_epochs_{args.num_epochs}-augment_{args.augment}-shuffle_{args.shuffle}-debug_data_skip_interval_{args.debug_data_skip_interval}-always_use_full_val_{args.always_use_full_val}-docs_encoder{args.docs_encoder}-code_encoder_{args.code_encoder}-num_gpus_{args.num_gpus}")

    trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, logger=logger, log_every_n_steps=10, flush_logs_every_n_steps=50, reload_dataloaders_every_n_epochs=1, accelerator=args.accelerator, plugins=args.plugins, precision=args.precision)

    trainer.fit(model)

if __name__ == "__main__":
    # [PARSE ARGUMENTS] (if they are given, otherwise keep default value)
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_encoder", type=str, default="microsoft/codebert-base")
    parser.add_argument("--code_encoder", type=str, default="microsoft/codebert-base")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--effective_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--effective_queue_size", type=int, default=64)
    parser.add_argument("--momentum_update_weight", type=float, default=0.999)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--base_data_folder", type=str, default="datasets/CodeSearchNet")
    parser.add_argument("--debug_data_skip_interval", type=int, default=100) # skips data during the loading process, which effectively makes us use a subset of the original data
    parser.add_argument("--always_use_full_val", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="dp")
    parser.add_argument("--plugins", type=str, default=None)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    print(f"[HYPERPARAMETERS] Hyperparameters: num_epochs={args.num_epochs}; effective_batch_size={args.effective_batch_size}; learning_rate={args.learning_rate}; temperature={args.temperature}; effective_queue_size={args.effective_queue_size}; momentum_update_weight={args.momentum_update_weight}; shuffle={args.shuffle}; augment={args.augment}; DEBUG_data_skip_interval={args.debug_data_skip_interval}; always_use_full_val={args.always_use_full_val}; base_data_folder={args.base_data_folder}; seed={args.seed}; num_workers={args.num_workers}, accelerator={args.accelerator}, plugins={args.plugins}, num_gpus={args.num_gpus}")

    execute(args)


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

# INSTALLED FROM https://github.com/ildoonet/pytorch-gradual-warmup-lr
from warmup_scheduler import GradualWarmupScheduler

from torch.optim.lr_scheduler import StepLR

class xMoCoModelPTL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.docs_tokenizer = None
        self.code_tokenizer = None

        self.docs_fast_encoder = AutoModel.from_pretrained(args.docs_encoder)
        self.docs_slow_encoder = AutoModel.from_pretrained(args.docs_encoder)
        self.code_fast_encoder = AutoModel.from_pretrained(args.code_encoder)
        self.code_slow_encoder = AutoModel.from_pretrained(args.code_encoder)

        self.effective_queue_size = args.effective_queue_size
        self.update_weight = args.momentum_update_weight
        self.batch_size = args.effective_batch_size
        self.temperature = args.temperature
        self.num_gpus = args.num_gpus

        if args.use_hard_negatives:
            #stuff needed for FAISS indexing
            self.raw_data=None
            self.negative_docs_queue = None
            self.negative_code_queue = None
            self.num_hard_negatives=args.num_hard_negatives

        # queues
        self.register_buffer("docs_queue", torch.randn(self.effective_queue_size, 768))
        self.register_buffer("code_queue", torch.randn(self.effective_queue_size, 768))

        self.docs_queue = F.normalize(self.docs_queue, p=2, dim=1)
        self.code_queue = F.normalize(self.code_queue, p=2, dim=1)

        # we actually only need one indices and current_index storage buffer, but that would make the replaceOldestQueueEntry implementation a bit nasty. Since these aren't big, I'll just keep it :P

        # dataset indices of the samples in the queues
        self.register_buffer("docs_indices", torch.empty(self.effective_queue_size).fill_(-1))
        self.register_buffer("code_indices", torch.empty(self.effective_queue_size).fill_(-1))

        # pointers to the starting index of the next block to be replaced
        self.register_buffer("docs_current_index", torch.zeros(1, dtype=torch.long))
        self.register_buffer("code_current_index", torch.zeros(1, dtype=torch.long))
        #labels for training: generating these every time and pushing them to gpu is apparently very slow, so do it here instead
        # self.register_buffer("labels", torch.zeros(int(self.batch_size/self.num_gpus), dtype=torch.long))

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
#        scheduler_steplr = StepLR(optimizer, step_size=1, gamma=np.exp(np.log(0.9)/(25000/self.batch_size))) #we want a decay of 0.9 per epoch
 #       scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=int(25000/self.batch_size), after_scheduler = scheduler_steplr) # notice the magic number for the number of total epochs
  #      return {"optimizer": optimizer, "lr_scheduler": scheduler_warmup}
        return optimizer

    def load_tokenizers(self):
        if self.docs_tokenizer == None:
            self.docs_tokenizer = AutoTokenizer.from_pretrained(self.args.docs_encoder)
        if self.code_tokenizer == None:
            self.code_tokenizer = AutoTokenizer.from_pretrained(self.args.code_encoder)

    @torch.no_grad()
    def train_dataloader(self):
        self.load_tokenizers()

        if self.args.use_hard_negatives:
            train_loader, self.raw_data = generateDataLoader(self.args.language, "train", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=self.args.shuffle, augment=self.args.augment, num_workers=self.args.num_workers)
            assert self.raw_data
            generateHardNegativeSearchIndices(self) # need to do this before every training epoch
        else:
            train_loader = generateDataLoader(self.args.language, "train", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=self.args.shuffle, augment=self.args.augment, num_workers=self.args.num_workers)

        return train_loader

    @torch.no_grad()
    def val_dataloader(self):
        self.load_tokenizers()
        self.val_loader = generateDataLoader(self.args.language, "valid", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=False, augment=False, num_workers=self.args.num_workers)
        return self.val_loader

    # ADAPTED FROM https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/moco/moco2_module.py#L137
    @torch.no_grad()
    def updateMomentumEncoder(self, fast_encoder, slow_encoder):
        """Momentum update of the key encoder."""
        for param_fast, param_slow in zip(fast_encoder.parameters(), slow_encoder.parameters()):
            em = self.update_weight
            param_slow.data = param_slow.data * em + param_fast.data * (1.0 - em)

    def replaceOldestQueueEntry(self, queue, queue_indices, current_index, newEntry, newIndices):

        gatheredNewEntries = self.concatAllGather(newEntry)
        gatheredNewIndices = self.concatAllGather(newIndices)

        pointer = int(current_index)
        queue[pointer : pointer+self.batch_size, :] = gatheredNewEntries.detach() # the queue doesn't need the computational graph
        queue_indices[pointer : pointer+self.batch_size] = gatheredNewIndices
        current_index[0] = (pointer + self.batch_size) % self.effective_queue_size

    def concatAllGather(self, tensor):
        if args.accelerator=="dp":
            return tensor
        gathered_tensors = self.all_gather(tensor)
        cache_tensor = torch.tensor([]).type_as(tensor)
        for elem in gathered_tensors:
            cache_tensor=torch.cat((cache_tensor, torch.squeeze(elem)), dim=0)
        return cache_tensor

    def forward(self, docs_samples, code_samples, isInference=False):
        if not isInference:
            docs_embeddings = self.docs_fast_encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
            code_embeddings = self.code_fast_encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
        else:
            with torch.no_grad():
                docs_embeddings = self.docs_fast_encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
                code_embeddings = self.code_fast_encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
        return docs_embeddings, code_embeddings

    def slow_forward(self, docs_samples, code_samples):
        with torch.no_grad():
            slow_docs_embeddings = self.docs_slow_encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
            slow_code_embeddings = self.code_slow_encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
        return slow_docs_embeddings, slow_code_embeddings

    def training_step(self, batch, batch_idx):

        # update FAISS index in-epoch
        #if self.args.use_hard_negatives and batch_idx % int(ceil(self.num_batches/2)+1) == 0 and batch_idx!=0:
        #    generateHardNegativeSearchIndices(self)

        docs_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}
        batch_indices = batch["index"]

        current_batch_size = docs_samples["input_ids"].shape[0]

        # [UPDATE MOMENTUM ENCODERS]
        self.updateMomentumEncoder(self.docs_fast_encoder, self.docs_slow_encoder)
        self.updateMomentumEncoder(self.code_fast_encoder, self.code_slow_encoder)

        # [FORWARD PASS] (compute embeddings using the fast encoders)
        docs_embeddings, code_embeddings = self(docs_samples, code_samples)
        positive_slow_docs_embeddings, positive_slow_code_embeddings = self.slow_forward(docs_samples, code_samples)

        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)
        positive_slow_docs_embeddings = F.normalize(positive_slow_docs_embeddings, p=2, dim=1)
        positive_slow_code_embeddings = F.normalize(positive_slow_code_embeddings, p=2, dim=1)

        l_pos_nl_pl = torch.matmul(docs_embeddings.view((current_batch_size, -1)), torch.transpose(positive_slow_code_embeddings, 0, 1))
        l_pos_pl_nl = torch.matmul(code_embeddings.view((current_batch_size, -1)), torch.transpose(positive_slow_docs_embeddings, 0, 1))

        l_neg_nl_pl = torch.matmul(docs_embeddings.view((current_batch_size, -1)), torch.transpose(self.code_queue, 0, 1))
        l_neg_pl_nl = torch.matmul(code_embeddings.view((current_batch_size, -1)), torch.transpose(self.docs_queue, 0, 1))

        logits_nl_pl = torch.cat((l_pos_nl_pl.view((current_batch_size, -1)), l_neg_nl_pl), dim=1)
        logits_pl_nl = torch.cat((l_pos_pl_nl.view((current_batch_size, -1)), l_neg_pl_nl), dim=1)

        # [FIND HARD NEGATIVES] (if enabled, also, this is only implemented in the non-MLP version for now)
        if args.use_hard_negatives:
            _, hard_negative_docs_indices = self.code_faiss.search(docs_embeddings.detach().cpu().numpy(), self.num_hard_negatives)
            _, hard_negative_code_indices = self.docs_faiss.search(code_embeddings.detach().cpu().numpy(), self.num_hard_negatives)

            #hard_negative_docs_embeddings = torch.tensor([]).type_as(self.negative_docs_queue)
            #hard_negative_code_embeddings = torch.tensor([]).type_as(self.negative_docs_queue)

            hard_negative_docs_samples = {"input_ids": torch.tensor([]).type_as(code_samples["input_ids"]), "attention_mask": torch.tensor([]).type_as(code_samples["attention_mask"])}
            hard_negative_code_samples = {"input_ids": torch.tensor([]).type_as(code_samples["input_ids"]), "attention_mask": torch.tensor([]).type_as(code_samples["attention_mask"])}

            for indices in hard_negative_docs_indices:
                hard_negative_sample = self.raw_data[indices]
                hard_negative_code_samples["input_ids"] = torch.cat((hard_negative_code_samples["input_ids"], hard_negative_sample["code_input_ids"].type_as(code_samples["input_ids"])), dim=0)
                hard_negative_code_samples["attention_mask"] = torch.cat((hard_negative_code_samples["attention_mask"], hard_negative_sample["code_attention_mask"].type_as(code_samples["attention_mask"])), dim=0)

            for indices in hard_negative_code_indices:
                hard_negative_sample = self.raw_data[indices]
                hard_negative_docs_samples["input_ids"] = torch.cat((hard_negative_docs_samples["input_ids"], hard_negative_sample["doc_input_ids"].type_as(code_samples["input_ids"])), dim=0)
                hard_negative_docs_samples["attention_mask"] = torch.cat((hard_negative_docs_samples["attention_mask"], hard_negative_sample["doc_attention_mask"].type_as(code_samples["attention_mask"])), dim=0)

            # THE APPROACH OF USING THE EMBEDDINGS COMPUTED DURING INDEX GENERATION SEEMS TO BE FAULTY! IT USES HIGHLY STALE DATA. INSTEAD, ALL OF THE FOUND NEGATIVE INDICES SHOULD BE FORWARDED THROUGH THE FAST ENCODER AND THEN MULTIPLIED WITH THE POSITIVE ENCODING
            # Note: Forwarding is a lot more expensive than  this approach, but if we only use a couple of hard negatives per positive element, the cost should be negligible (e.g. k=3)
            # also note that the cost of the forwarding grows really quickly in the number of harde negative sampels and batch size (i.e. O(bs*k)), but the batch/matrix multiplication for computing the actual similarities is basically free in comparison. Same thing holds for filtering false negatives. I initially thought that this would be really expensive, but it turns out it's only expensive for the queue because it is so large. it should always be applied to the hard negative similarity matrix

            #for indices in hard_negative_docs_indices:
            #    hard_negative_docs_embeddings = torch.cat((hard_negative_docs_embeddings, torch.unsqueeze(self.negative_docs_queue[indices], dim=0)), dim=0)
            #for indices in hard_negative_code_indices:
            #    hard_negative_code_embeddings = torch.cat((hard_negative_code_embeddings, torch.unsqueeze(self.negative_code_queue[indices], dim=0)), dim=0)
        # [COMPUTE LOSS DIRECTLY ON ENCODER OUTPUT]

        # compute similarity of positive fast_NL/slow_PL pairs

            #l_hard_neg_nl_pl = torch.bmm(hard_negative_docs_embeddings.view((current_batch_size, self.num_hard_negatives, 768)), docs_embeddings.view((current_batch_size, 768, 1)))
            #l_hard_neg_pl_nl = torch.bmm(hard_negative_code_embeddings.view((current_batch_size, self.num_hard_negatives, 768)), code_embeddings.view((current_batch_size, 768, 1)))

            hard_negative_docs_embeddings, hard_negative_code_embeddings = self(hard_negative_docs_samples, hard_negative_code_samples, isInference=True)
            hard_negative_docs_embeddings = F.normalize(hard_negative_docs_embeddings, p=2, dim=1)
            hard_negative_code_embeddings = F.normalize(hard_negative_code_embeddings, p=2, dim=1)

            l_hard_neg_nl_pl = torch.matmul(docs_embeddings.view((current_batch_size, 768)), torch.transpose(hard_negative_docs_embeddings.view((-1, 768)), 0, 1))
            l_hard_neg_pl_nl = torch.matmul(code_embeddings.view((current_batch_size, 768)), torch.transpose(hard_negative_code_embeddings.view((-1, 768)), 0, 1))

            l_hard_neg_nl_pl = torch.squeeze(l_hard_neg_nl_pl)
            l_hard_neg_pl_nl = torch.squeeze(l_hard_neg_pl_nl)

            # mask out all false negatives in the nl_pl part
            hard_negative_docs_indices = hard_negative_docs_indices.reshape(-1)
            for i in range(current_batch_size):
                for j in range(hard_negative_docs_indices.shape[0]):
                    if hard_negative_docs_indices[j] == batch_indices[i]:
                        l_hard_neg_nl_pl[i][j] = -1

            #mask out all false negatives in the pl_nl part
            hard_negative_code_indices = hard_negative_code_indices.reshape(-1)
            for i in range(current_batch_size):
                for j in range(hard_negative_code_indices.shape[0]):
                    if hard_negative_code_indices[j] == batch_indices[i]:
                        l_hard_neg_pl_nl[i][j] = -1

        #if self.args.remove_duplicates: # extremely computationally expensive
            # mask out any logits entry for which the queue entry is a duplicate of the positive sample
        #    masking_list = []#[[target for target, index in enumerate(line) if index==pos_index] for ]#[index for index, x in enumerate(batch_indices) if x in self.code_indices]
        #    for k in range(current_batch_size):
        #        pos_index = batch_indices[k]
        #        cache_list = []
        #        for i, line in enumerate(self.docs_indices):
        #            if line==pos_index:
        #                cache_list.append(i)
        #        masking_list.append(cache_list)

            # print("masking list generated...")
        #    for i, mask in enumerate(masking_list):
        #        l_neg_nl_pl[i, mask]=-1
        #        l_neg_pl_nl[i, mask]=-1

            logits_nl_pl = torch.cat((logits_nl_pl, l_hard_neg_nl_pl), dim=1)
            logits_pl_nl = torch.cat((logits_pl_nl, l_hard_neg_pl_nl), dim=1)

        labels = torch.tensor(range(logits_nl_pl.shape[0])).type_as(code_samples["input_ids"])

        loss_nl_pl = nn.CrossEntropyLoss()(logits_nl_pl/self.temperature, labels)

        loss_pl_nl = nn.CrossEntropyLoss()(logits_pl_nl/self.temperature, labels)

        loss = (loss_nl_pl + loss_pl_nl) / 2.0

        # [LOGGING]
        self.log("Loss/training", loss.item(), sync_dist=True)

        # [UPDATE THE QUEUES]
        self.replaceOldestQueueEntry(self.docs_queue, self.docs_indices, self.docs_current_index, positive_slow_docs_embeddings, batch_indices)
        self.replaceOldestQueueEntry(self.code_queue, self.code_indices, self.code_current_index, positive_slow_code_embeddings, batch_indices)

        # [UPDAE THE LEARNING RATE]
        #self.lr_schedulers().step()

        return loss

    def validation_step(self, batch, batch_idx):
        docs_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}

        docs_embeddings, code_embeddings = self(docs_samples, code_samples, isInference=True)

        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        return {"index": batch["index"].view(-1), "docs_embeddings": docs_embeddings, "code_embeddings": code_embeddings}

    def validation_step_end(self, batch_parts): # I know this seems unnecessary at first glance, but the original function would only return the very first entry of every tensor in the dictionary. But we need all entries
        return batch_parts

    def validation_epoch_end(self, outputs):
        code_emb_list = torch.tensor([]).type_as(outputs[0]["docs_embeddings"])
        docs_emb_list = torch.tensor([]).type_as(outputs[0]["docs_embeddings"])

        for output in outputs:
            code_emb_list = torch.cat((code_emb_list, output["code_embeddings"]), dim=0)
            docs_emb_list = torch.cat((docs_emb_list, output["docs_embeddings"]), dim=0)

        code_emb_list = self.concatAllGather(code_emb_list)

        local_rank = self.global_rank

        basis_index = docs_emb_list.shape[0]
        # labels are on a "shifted" diagonal
        labels = torch.tensor([x for x in range(basis_index * local_rank, basis_index * (local_rank + 1))]).type_as(outputs[0]["docs_embeddings"])

        assert (docs_emb_list.shape[1] == 768)
        validation_computations(self, docs_emb_list, code_emb_list, labels, "Accuracy_enc/validation", "Similarity_enc", substring="ENC")

# COPIED FROM https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/codesearch/run_classifier.py#L45
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed) # I don't use numpy, but some library I use here might use numpy as a backend
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def execute(args):

    import os
    os.environ["TOKENIZERS_PARALLELISM"]="false"

    if args.use_hard_negatives:
        os.environ["OMP_WAIT_POLICY"]="PASSIVE"
        global generateHardNegativeSearchIndices
        from utils.hard_negative_search import generateHardNegativeSearchIndices

    # set all seeds so we can ensure same MLP initialization and augmentation behaviour on all GPUS
    set_seed(args.seed)

    model = xMoCoModelPTL(args)

    now = datetime.now()
    now_str = now.strftime("%b%d_%H_%M_%S")
    logger = pl.loggers.TensorBoardLogger("runs", name=f"{now_str}-xMoCo-language_{args.language}-eff_bs_{args.effective_batch_size}-lr_{args.learning_rate}-eff_qs_{args.effective_queue_size}-max_epochs_{args.num_epochs}-aug_{args.augment}-shuf_{args.shuffle}-debug_skip_interval_{args.debug_data_skip_interval}-always_full_val_{args.always_use_full_val}-docs_enc_{args.docs_encoder}-code_enc_{args.code_encoder}-num_gpus_{args.num_gpus}-rmv_dup_{args.remove_duplicates}-use_hard_negatives_{args.use_hard_negatives}-num_hard_negatives_{0 if not args.use_hard_negatives else args.num_hard_negatives}")

    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval="step")


    trainer = pl.Trainer(callbacks=[lr_monitor], gpus=args.num_gpus, max_epochs=args.num_epochs, logger=logger, log_every_n_steps=10, flush_logs_every_n_steps=50, reload_dataloaders_every_n_epochs=1, accelerator=args.accelerator, plugins=args.plugins, precision=args.precision)

#    from pytorch_lightning.plugins import DDPPlugin
    #trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, logger=logger, reload_dataloaders_every_n_epochs=1, plugins="deepspeed_stage_2", precision=args.precision)
    #trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, logger=logger, reload_dataloaders_every_n_epochs=1, plugins=DDPPlugin(find_unused_parameters=False))

#    trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, logger=logger, reload_dataloaders_every_n_epochs=1, accelerator=args.accelerator, plugins=args.plugins)
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
    parser.add_argument("--effective_queue_size", type=int, default=128)
    parser.add_argument("--momentum_update_weight", type=float, default=0.999)
    parser.add_argument("--shuffle", action="store_true", default=False) # note: when using dp or ddp, the value of this will be ignored and the dataset will always be shuffled (would need to make a custom DistributedSampler to prevent this)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--base_data_folder", type=str, default="/itet-stor/jstuder/net_scratch/nl-pl_moco/datasets/CodeSearchNet")
    parser.add_argument("--debug_data_skip_interval", type=int, default=1) # skips data during the loading process, which effectively makes us use a subset of the original data
    parser.add_argument("--always_use_full_val", action="store_true", default=False)
    parser.add_argument("--remove_duplicates", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="ddp")
    parser.add_argument("--plugins", type=str, default=None)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--language", type=str, default="ruby")
    parser.add_argument("--enable_mlp", action="store_true", default=False) # this currently does nothing
    parser.add_argument("--use_hard_negatives", action="store_true", default=False)
    parser.add_argument("--num_hard_negatives", type=int, default=4)
    args = parser.parse_args()

    print(f"[HYPERPARAMETERS] Hyperparameters: xMoCo - language={args.language} - num_epochs={args.num_epochs}; effective_batch_size={args.effective_batch_size}; learning_rate={args.learning_rate}; temperature={args.temperature}; effective_queue_size={args.effective_queue_size}; momentum_update_weight={args.momentum_update_weight}; shuffle={args.shuffle}; augment={args.augment}; DEBUG_data_skip_interval={args.debug_data_skip_interval}; always_use_full_val={args.always_use_full_val}; base_data_folder={args.base_data_folder}; seed={args.seed}; num_workers={args.num_workers}, accelerator={args.accelerator}, plugins={args.plugins}, num_gpus={args.num_gpus}, remove_duplicates={args.remove_duplicates}, language={args.language}, enable_mlp={args.enable_mlp}, use_hard_negatives={args.use_hard_negatives}, num_hard_negatives={0 if not args.use_hard_negatives else args.num_hard_negatives}")

    execute(args)


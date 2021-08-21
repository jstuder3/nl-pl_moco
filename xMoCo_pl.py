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
        self.docs_slow_encoder = AutoModel.from_pretrained(args.docs_encoder)
        self.code_fast_encoder = AutoModel.from_pretrained(args.code_encoder)
        self.code_slow_encoder = AutoModel.from_pretrained(args.code_encoder)

        if args.enable_mlp:
            self.docs_fast_mlp = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 128))
            self.docs_slow_mlp = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 128))
            self.code_fast_mlp = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 128))
            self.code_slow_mlp = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 128))

        self.effective_queue_size = args.effective_queue_size
        self.update_weight = args.momentum_update_weight
        self.batch_size = args.effective_batch_size
        self.temperature = args.temperature
        self.num_gpus = args.num_gpus

        # queues
        self.register_buffer("docs_queue", torch.randn(self.effective_queue_size, 768))
        self.register_buffer("code_queue", torch.randn(self.effective_queue_size, 768))

        # we actually only need one indices and current_index storage buffer, but that would make the replaceOldestQueueEntry implementation a bit nasty. Since these aren't big, I'll just keep it :P

        # dataset indices of the samples in the queues
        self.register_buffer("docs_indices", torch.empty(self.effective_queue_size).fill_(-1))
        self.register_buffer("code_indices", torch.empty(self.effective_queue_size).fill_(-1))

        # pointers to the starting index of the next block to be replaced
        self.register_buffer("docs_current_index", torch.zeros(1, dtype=torch.long))
        self.register_buffer("code_current_index", torch.zeros(1, dtype=torch.long))

        self.save_hyperparameters()

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
        train_loader = generateDataLoader(self.args.language, "train", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=self.args.shuffle, augment=self.args.augment, num_workers=self.args.num_workers)
        return train_loader

    @torch.no_grad()
    def val_dataloader(self):
        self.load_tokenizers()
        val_loader = generateDataLoader(self.args.language, "valid", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=False, augment=False, num_workers=self.args.num_workers)
        return val_loader

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

    def mlp_foward(self, positive_fast_encodings, positive_slow_encodings, queue_encodings, fast_mlp, slow_mlp):#, additional_samples=None): # can we possibly use this in combination with the hard negatives? #assumption for now: this probably won't work
        assert self.args.enable_mlp, "Assertion failed: Called mlp_forward while enable_mlp flag was not set"

        positive_mlp_encodings = fast_mlp(positive_fast_encodings)
        positive_slow_mlp_encodings = slow_mlp(positive_slow_encodings)
        queue_mlp_encodings = slow_mlp(queue_encodings)

        #if additional_samples:
        #    additional_encodings = slow_mlp(additional_samples) # slow or fast here for the hard negatives? either one seems wrong because the hard negatives are probably already stale
        #    return positive_mlp_encodings, queue_mlp_encodings, additional_encodings
        return positive_mlp_encodings, positive_slow_mlp_encodings, queue_mlp_encodings

    def training_step(self, batch, batch_idx):
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

        if self.args.enable_mlp:
            # [MLP FORWARD PASS]
            docs_mlp_embeddings, positive_mlp_docs_embeddings, docs_queue_mlp_embeddings = self.mlp_forward(docs_embeddings, positive_slow_docs_embeddings, self.docs_queue, self.docs_fast_mlp, self.docs_slow_mlp)
            code_mlp_embeddings, positive_mlp_code_embeddings, code_queue_mlp_embeddings = self.mlp_forward(code_embeddings, positive_slow_code_embeddings, self.code_queue, self.code_fast_mlp, self.code_slow_mlp)

            # [COMPUTE LOSS ON MLP OUPTUTS]
            l_pos_nl_pl = torch.bmm(docs_mlp_embeddings.view((current_batch_size, 1, -1)), positive_mlp_code_embeddings.view((current_batch_size, -1, 1))) # compute similarity between positive fast_docs/slow_code pairs
            l_pos_pl_nl = torch.bmm(code_mlp_embeddings.view((current_batch_size, 1, -1)), positive_mlp_docs_embeddings.view((current_batch_size, -1, 1))) # compute similarity between positive fast_code/slow_docs pairs

            l_neg_nl_pl = torch.matmul(docs_mlp_embeddings.view((current_batch_size, -1)), torch.transpose(code_queue_mlp_embeddings))
            l_neg_pl_nl = torch.matmul(code_mlp_embeddings.view((current_batch_size, -1)), torch.transpose(docs_queue_mlp_embeddings))

        else:
            # [COMPUTE LOSS DIRECTLY ON ENCODER OUTPUT]

            # compute similarity of positive fast_NL/slow_PL pairs
            l_pos_nl_pl = torch.bmm(docs_embeddings.view((current_batch_size, 1, -1)), positive_slow_code_embeddings.view((current_batch_size, -1, 1)))
            l_pos_pl_nl = torch.bmm(code_embeddings.view((current_batch_size, 1, -1)), positive_slow_docs_embeddings.view((current_batch_size, -1, 1)))

            l_neg_nl_pl = torch.matmul(docs_embeddings.view((current_batch_size, -1)), torch.transpose(self.code_queue, 0, 1))
            l_neg_pl_nl = torch.matmul(code_embeddings.view((current_batch_size, -1)), torch.transpose(self.docs_queue, 0, 1))

        logits_nl_pl = torch.cat((l_pos_nl_pl.view((current_batch_size, 1)), l_neg_nl_pl), dim=1)
        logits_pl_nl = torch.cat((l_pos_pl_nl.view((current_batch_size, 1)), l_neg_pl_nl), dim=1)

        if not self.args.dont_remove_duplicates: # extremely computationally expensive
            # remove/mask out any logits entry for which the queue contained a duplicate
            inclusion_list = torch.tensor([index for index, x in enumerate(batch_indices) if x not in self.code_indices]).type_as(logits_nl_pl).long()
            logits_nl_pl=logits_nl_pl[inclusion_list]
            logits_pl_nl=logits_pl_nl[inclusion_list]

        # labels: the entries from l_pos should always contain the smallest values
        labels = torch.tensor([0 for _ in range(logits_nl_pl.shape[0])]).type_as(code_samples["input_ids"])
        loss_nl_pl = nn.CrossEntropyLoss()(logits_nl_pl/self.temperature, labels)

        loss_pl_nl = nn.CrossEntropyLoss()(logits_pl_nl/self.temperature, labels)

        loss = (loss_nl_pl + loss_pl_nl) / 2.0

        self.log("Loss/training", loss.item(), sync_dist=True)

        #print(f"Loss on GPU {self.global_rank}: {loss.item()} with loss_nl_pl {loss_nl_pl.item()}, loss_pl_nl {loss_pl_nl.item()}")

        # [UPDATE THE QUEUES]
        #negative_slow_docs_embeddings, negative_slow_code_embeddings = self.slow_forward(docs_samples, code_samples)
        self.replaceOldestQueueEntry(self.docs_queue, self.docs_indices, self.docs_current_index, positive_slow_docs_embeddings, batch_indices)
        self.replaceOldestQueueEntry(self.code_queue, self.code_indices, self.code_current_index, positive_slow_code_embeddings, batch_indices)

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
    # set all seeds so we can ensure same MLP initialization and augmentation behaviour on all GPUS
    set_seed(args.seed)

    model = xMoCoModelPTL(args)

    now = datetime.now()
    now_str = now.strftime("%b%d_%H_%M_%S")
    logger = pl.loggers.TensorBoardLogger("runs", name=f"{now_str}-xMoCo-eff_bs_{args.effective_batch_size}-lr_{args.learning_rate}-eff_qs_{args.effective_queue_size}-max_epochs_{args.num_epochs}-aug_{args.augment}-shuf_{args.shuffle}-debug_skip_interval_{args.debug_data_skip_interval}-always_full_val_{args.always_use_full_val}-docs_enc_{args.docs_encoder}-code_enc_{args.code_encoder}-num_gpus_{args.num_gpus}; no_rmv_dup_{args.dont_remove_duplicates}")

    trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, logger=logger, log_every_n_steps=10, flush_logs_every_n_steps=50, reload_dataloaders_every_n_epochs=1, accelerator=args.accelerator, plugins=args.plugins, precision=args.precision)

    trainer.fit(model)

if __name__ == "__main__":
    # [PARSE ARGUMENTS] (if they are given, otherwise keep default value)
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_encoder", type=str, default="microsoft/codebert-base")
    parser.add_argument("--code_encoder", type=str, default="microsoft/codebert-base")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--effective_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--effective_queue_size", type=int, default=64)
    parser.add_argument("--momentum_update_weight", type=float, default=0.999)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--base_data_folder", type=str, default="datasets/CodeSearchNet")
    parser.add_argument("--debug_data_skip_interval", type=int, default=100) # skips data during the loading process, which effectively makes us use a subset of the original data
    parser.add_argument("--always_use_full_val", action="store_true", default=False)
    parser.add_argument("--dont_remove_duplicates", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="dp")
    parser.add_argument("--plugins", type=str, default=None)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--enable_mlp", action="store_true", default=False) # it's very much possible that this will suck up an incredible amount of gpu memory depending on the queue size, so use with caution
    args = parser.parse_args()

    print(f"[HYPERPARAMETERS] Hyperparameters: xMoCo - num_epochs={args.num_epochs}; effective_batch_size={args.effective_batch_size}; learning_rate={args.learning_rate}; temperature={args.temperature}; effective_queue_size={args.effective_queue_size}; momentum_update_weight={args.momentum_update_weight}; shuffle={args.shuffle}; augment={args.augment}; DEBUG_data_skip_interval={args.debug_data_skip_interval}; always_use_full_val={args.always_use_full_val}; base_data_folder={args.base_data_folder}; seed={args.seed}; num_workers={args.num_workers}, accelerator={args.accelerator}, plugins={args.plugins}, num_gpus={args.num_gpus}, dont_remove_duplicates={args.dont_remove_duplicates}, language={args.language}, enable_mlp={args.enable_mlp}")

    execute(args)

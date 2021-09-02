# my implementation of DyHardCode for comparison and verification purposes

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
import argparse
import random
from datetime import datetime
import numpy as np
import os

import time
from math import floor, ceil

from utils.improved_data_loading import generateDataLoader
from utils.metric_computation import validation_computations
from utils.single_modality_hard_negative_search import generateHardNegativeSearchIndices

os.environ["TOKENIZERS_PARALLELISM"]="false"

class DyHardCodeModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args=args

        self.tokenizer = None
        self.encoder = AutoModel.from_pretrained(args.model_name)

        self.effective_queue_size = args.effective_queue_size
        self.update_weight = args.momentum_update_weight
        self.batch_size = args.effective_batch_size
        self.temperature = args.temperature
        self.num_gpus = args.num_gpus

        self.raw_data=None
        self.negative_matrix = None
        self.num_hard_negatives=args.num_hard_negatives

    def configure_optimizers(self):
        if self.global_rank == 0:
            # compute lowest theoretically achievable loss as a sanity check
            # the lowest loss is achieved when the positive has value 1 (cosine similarity, identical embedding) and all others point in the opposite direction (cosine similarity of -1), assuming the matmul version of hard negatives
            lowest_loss = -np.log(np.e/(np.e+((1+self.num_hard_negatives) * self.batch_size/self.num_gpus - 1)*np.exp(-1)))
            print(f"Lowest theoretically achievable loss with local batch size of {self.batch_size/self.num_gpus} and {self.num_hard_negatives} hard negatives per element (assuming matmul version): {lowest_loss:.8f}")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def train_dataloader(self):
        if self.tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if args.use_hard_negatives:
            train_loader, self.raw_data = generateDataLoader(self.args.language, "train", self.tokenizer, self.args, batch_size=int(self.args.effective_batch_size/self.args.num_gpus), shuffle=self.args.shuffle, augment=self.args.augment, num_workers=self.args.num_workers)  # int(math.floor(multiprocessing.cpu_count()/torch.cuda.device_count())))
            generateHardNegativeSearchIndices(self)
        else:
            train_loader = generateDataLoader(self.args.language, "train", self.tokenizer, self.args, batch_size=int(self.args.effective_batch_size/self.args.num_gpus), shuffle=self.args.shuffle, augment=self.args.augment, num_workers=self.args.num_workers)  # int(math.floor(multiprocessing.cpu_count()/torch.cuda.device_count())))

        self.num_batches = int(floor(len(train_loader)/self.num_gpus))
        return train_loader

    def val_dataloader(self):
        if self.tokenizer==None:
            self.tokenizer=AutoTokenizer.from_pretrained(self.args.model_name)
        val_loader = generateDataLoader(self.args.language, "valid", self.tokenizer, self.args, batch_size=int(self.args.effective_batch_size/self.args.num_gpus), shuffle=False, augment=False, num_workers=self.args.num_workers)#int(math.floor(multiprocessing.cpu_count()/torch.cuda.device_count())))
        return val_loader

    def concatAllGather(self, tensor):
        if args.accelerator == "dp":
            return tensor
        gathered_tensors = self.all_gather(tensor)
        cache_tensor = torch.tensor([]).type_as(tensor)
        for elem in gathered_tensors:
            cache_tensor = torch.cat((cache_tensor, torch.squeeze(elem)), dim=0)
        return cache_tensor

    def forward(self, docs_samples, code_samples, isInference=False):
        if not isInference:
            docs_embeddings = self.encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
            code_embeddings = self.encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
        else:
            with torch.no_grad():
                docs_embeddings = self.encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
                code_embeddings = self.encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
        return docs_embeddings, code_embeddings

    def training_step(self, batch, batch_idx):

        # update FAISS index in-epoch
        #if self.args.use_hard_negatives and batch_idx % int(ceil(self.num_batches/2)+1) == 0 and batch_idx!=0:
        #    generateHardNegativeSearchIndices(self)

        docs_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}
        batch_indices = batch["index"].cpu() #we only ever use the indices on the cpu, so we remove a lot of overhead by not having to fetch it from the gpu every time

        current_batch_size = docs_samples["input_ids"].shape[0]

        docs_embeddings, code_embeddings = self(docs_samples, code_samples)

        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        l_pos = torch.matmul(docs_embeddings, torch.transpose(code_embeddings, 0, 1))

        if self.args.use_hard_negatives:
            #st = time.time()
            _, hard_negative_docs_indices = self.faiss_index.search(docs_embeddings.detach().cpu().numpy(), self.num_hard_negatives)
            #print(f"Mining on gpu {self.global_rank} took {time.time()-st:.6f} seconds")
            k = self.num_hard_negatives
            #st = time.time()
            hard_negative_code_samples = {"input_ids": torch.tensor([]).type_as(code_samples["input_ids"]), "attention_mask": torch.tensor([]).type_as(code_samples["attention_mask"])}
            #hard_negative_code_samples = {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([])}


            for indices in hard_negative_docs_indices:
                hard_negative_sample = self.raw_data[indices]
                hard_negative_code_samples["input_ids"] = torch.cat((hard_negative_code_samples["input_ids"], hard_negative_sample["code_input_ids"].type_as(code_samples["input_ids"])), dim=0)
                hard_negative_code_samples["attention_mask"] = torch.cat((hard_negative_code_samples["attention_mask"], hard_negative_sample["code_attention_mask"].type_as(code_samples["attention_mask"])), dim=0)
                #hard_negative_code_samples["input_ids"] = torch.cat((hard_negative_code_samples["input_ids"], hard_negative_sample["code_input_ids"]), dim=0)
                #hard_negative_code_samples["attention_mask"] = torch.cat((hard_negative_code_samples["attention_mask"], hard_negative_sample["code_attention_mask"]), dim=0)

            #print(f"Data concatenation took {time.time()-st} seconds")

            #st = time.time()
            with torch.no_grad():
                hard_negative_code_embeddings = self.encoder(input_ids=hard_negative_code_samples["input_ids"], attention_mask=hard_negative_code_samples["attention_mask"])["pooler_output"]
                hard_negative_code_embeddings = F.normalize(hard_negative_code_embeddings, p=2, dim=1)
                #print(f"Hard negative forward pass took {time.time()-st} seconds")
                st=time.time()
            
                #hard_negative_docs_similarities = torch.bmm(hard_negative_code_embeddings.view((current_batch_size, self.num_hard_negatives, 768)), docs_embeddings.view((current_batch_size, 768, 1)))
                hard_negative_docs_similarities = torch.matmul(docs_embeddings, torch.transpose(hard_negative_code_embeddings, 0, 1))
                hard_negative_docs_similarities = torch.squeeze(hard_negative_docs_similarities)
            
            #print(f"BMM took {time.time()-st} seconds")
            
            #st = time.time()
            #hard_negative_docs_similarities=hard_negative_docs_similarities
            #print(f"Moving to CPU took {time.time()-st} seconds")

            #st = time.time()
            #remove false negatives (bmm version)
            #for i in range(current_batch_size):
            #    for j in range(k):
            #        if hard_negative_docs_indices[i][j] == batch_indices[i]:
            #            hard_negative_docs_similarities[i][j]=-1
            #print(f"False negative filtering took {time.time()-st} seconds")
            
            #remove false negatives (matmul version)
            hard_negative_docs_indices = hard_negative_docs_indices.reshape(-1)
            for i in range(current_batch_size):
                for j in range(hard_negative_docs_indices.shape[0]):
                    if hard_negative_docs_indices[j] == batch_indices[i]:
                        hard_negative_docs_similarities[i][j]=-1

            l_neg = hard_negative_docs_similarities#torch.from_numpy(hard_negative_docs_similarities).type_as(docs_embeddings)#hard_negative_docs_similarities.type_as(docs_embeddings) # this has to be of type float32
            logits = torch.cat((l_pos, l_neg), dim=1)

        else:
            logits = l_pos

        labels = torch.tensor(range(docs_embeddings.shape[0])).type_as(docs_samples["input_ids"]) #this has to be of type long

        loss = nn.CrossEntropyLoss()(logits/self.temperature, labels)

        self.log("Loss/training", loss.item(), sync_dist=True)

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
        labels = torch.tensor(range(basis_index * local_rank, basis_index * (local_rank + 1))).type_as(outputs[0]["docs_embeddings"])

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
    set_seed(args.seed)

    model = DyHardCodeModel(args)

    now = datetime.now()
    now_str = now.strftime("%b%d_%H_%M_%S")
    logger = pl.loggers.TensorBoardLogger("runs", name=f"{now_str}-DyHardCode-language_{args.language}-eff_bs_{args.effective_batch_size}-lr_{args.learning_rate}-max_epochs_{args.num_epochs}-aug_{args.augment}-shuf_{args.shuffle}-debug_skip_interval_{args.debug_data_skip_interval}-always_full_val_{args.always_use_full_val}-docs_enc_{args.model_name}-num_gpus_{args.num_gpus}-use_hard_negatives_{args.use_hard_negatives}-num_hard_negatives_{0 if not args.use_hard_negatives else args.num_hard_negatives}")

    trainer = pl.Trainer(log_gpu_memory="all", val_check_interval=0.5, gpus=args.num_gpus, max_epochs=args.num_epochs, logger=logger, log_every_n_steps=10, flush_logs_every_n_steps=50, reload_dataloaders_every_n_epochs=1, accelerator=args.accelerator, plugins=args.plugins, precision=args.precision)
#    from pytorch_lightning.plugins import DDPPlugin

#    trainer = pl.Trainer(log_gpu_memory="all", gpus=args.num_gpus, max_epochs=args.num_epochs, logger=logger, log_every_n_steps=10, flush_logs_every_n_steps=50, reload_dataloaders_every_n_epochs=1, accelerator=args.accelerator, plugins=DDPPlugin(find_unused_parameters=False), precision=args.precision)

    trainer.fit(model)

if __name__ == "__main__":
    # [PARSE ARGUMENTS] (if they are given, otherwise keep default value)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--effective_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--effective_queue_size", type=int, default=64)
    parser.add_argument("--momentum_update_weight", type=float, default=0.999)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--disable_normalizing_encoder_embeddings_during_training", action="store_true", default=False) # used for some experiments. disabling normalization of the encoder embeddings during training will destroy performance!
    parser.add_argument("--disable_mlp", action="store_true", default=False) # this is a bad idea and will probably lower the performance. super unintuitive though.
    parser.add_argument("--base_data_folder", type=str, default="/itet-stor/jstuder/net_scratch/nl-pl_moco/datasets/CodeSearchNet")
    parser.add_argument("--debug_data_skip_interval", type=int, default=100) # skips data during the loading process, which effectively makes us use a subset of the original data
    parser.add_argument("--always_use_full_val", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="ddp")
    parser.add_argument("--plugins", type=str, default=None)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--language", type=str, default="ruby")
    parser.add_argument("--use_hard_negatives", action="store_true", default=False)
    parser.add_argument("--num_hard_negatives", type=int, default=4)
    args = parser.parse_args()

    print(f"[HYPERPARAMETERS] Hyperparameters: DyHardCode - language={args.language} - num_epochs={args.num_epochs}; effective_batch_size={args.effective_batch_size}; learning_rate={args.learning_rate}; temperature={args.temperature}; effective_queue_size={args.effective_queue_size}; momentum_update_weight={args.momentum_update_weight}; shuffle={args.shuffle}; augment={args.augment}; DEBUG_data_skip_interval={args.debug_data_skip_interval}; always_use_full_val={args.always_use_full_val}; base_data_folder={args.base_data_folder}; disable_normalizing_encoder_embeddings_during_training={args.disable_normalizing_encoder_embeddings_during_training}; disable_mlp={args.disable_mlp}; seed={args.seed}; num_workers={args.num_workers}, accelerator={args.accelerator}, plugins={args.plugins}, num_gpus={args.num_gpus}")

    execute(args)

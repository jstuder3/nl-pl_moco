import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
import argparse
import random
from datetime import datetime
import time
import numpy as np
import multiprocessing
import math
import platform
import os

os.environ["TOKENIZERS_PARALLELISM"]="false"

import torch.distributed as dist

from utils.improved_data_loading import generateDataLoader

class MoCoModelPTL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.encoder = AutoModel.from_pretrained(self.args.model_name)
        self.momentum_encoder = AutoModel.from_pretrained(self.args.model_name)

        self.max_queue_size = self.args.max_queue_size
        self.update_weight = self.args.momentum_update_weight
        self.batch_size = self.args.batch_size
        self.num_gpus = torch.cuda.device_count()

        #self.queue = [] # this causes issues in multi-gpu settings, so we need to replace it with a register_buffer
        self.register_buffer("queue", torch.randn(self.max_queue_size*self.batch_size*torch.cuda.device_count(), 768)) # queue has dimensions [queue_size*batch_size, 768]. we initialize it with random numbers to prevent fitting entries which wouldn't actually contain anything yet if we were to use a list
        self.register_buffer("current_index", torch.zeros(1, dtype=torch.long))
        #self.current_index = 0 # for good measure, this is also replaced with a register_buffer

        if not self.args.disable_mlp:
            # 768 is output size of CodeBERT (i.e. BERT_base), 2048 is the hidden layer size MoCoV2 uses and 128 is the output size that SimCLR uses
            self.encoder_mlp = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 128))
            self.momentum_encoder_mlp = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 128))

        self.save_hyperparameters() # makes the hyperparameters available everywhere (but we don't care about that) and automatically logs them to tensorboard

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_loader = generateDataLoader("python", "train", self.tokenizer, self.args, shuffle=self.args.shuffle, augment=self.args.augment, num_workers=self.args.num_workers)#int(math.floor(multiprocessing.cpu_count()/torch.cuda.device_count())))
        return train_loader

    def val_dataloader(self):
        val_loader = generateDataLoader("python", "valid", self.tokenizer, self.args, shuffle=False, augment=False, num_workers=self.args.num_workers)#int(math.floor(multiprocessing.cpu_count()/torch.cuda.device_count())))
        return val_loader

    def update_momentum_encoder(self):
        # update momentum_encoder weights by taking the weighted average of the current weights and the new encoder weights
        # note: need to make sure that this actually works (update: seems to work)
        encoder_params = self.encoder.state_dict()
        for name, param in self.momentum_encoder.named_parameters():
            param = self.update_weight * param + (1 - self.update_weight) * encoder_params[name]

    def replaceOldestQueueEntry(self, newEntry):

        # DIFFERENCE TO SINGLE-GPU IMPLEMENTATION: NEED TO GATHER THE CODE EMBEDDINGS OF ALL OTHER GPUS OF THIS ITERATION

        gatheredNewEntries = self.concatAllGather(newEntry) # this could be problematic; need to check that this returns a tenosr of shape [batch_size, 768] instead of [batch_size/num_gpus, 768]

        # this function will replace the oldest ("most stale") entry of the queue
        pointer = int(self.current_index)
        self.queue[pointer : pointer+self.batch_size*torch.cuda.device_count(), :] = gatheredNewEntries.detach()  # we detach to make sure that we don't waste memory
        self.current_index[0] = (self.current_index+self.batch_size*torch.cuda.device_count()) % (self.max_queue_size*self.batch_size*torch.cuda.device_count()) #current_index should point at the beginning of a "block" of size batch_size

    def concatAllQueueEntries(self):
        #concat_tensor = torch.tensor([]).cuda()
        #for queue_entry in self.queue:
        #    concat_tensor = torch.cat((concat_tensor, queue_entry), dim=0)
        return self.queue # the register_buffer version is already concatenated
        #return concat_tensor

    def concatAllGather(self, tensor):
        gathered_tensors = self.all_gather(tensor)
       # print(f"concatAllGather called on {self.global_rank} with tensor of shape {tensor.shape} and gathered tensor shape of {gathered_tensors.shape}")
        #gathered_tensors = torch.cat(gathered_tensors, dim=0)
        cache_tensor = torch.tensor([]).type_as(tensor)
        for elem in gathered_tensors:
            cache_tensor=torch.cat((cache_tensor, torch.squeeze(elem)), dim=0)
       # print(f"concatAllGather called on {self.global_rank} with tensor of shape {tensor.shape} and post-concat cache_tensor shape of {cache_tensor.shape}")
        return cache_tensor
       # tensor_list = [tensor.ones_like(tensor) for _ in range(torch.cuda.device_count())]
       # torch.distributed.all_gather(tensor_list, tensor)
       # tensors_gather = torch.cat(tensor_list, dim=0)
       # return tensors_gather

    def forward(self, encoder_input, momentum_encoder_input, isInference=False):
        # note entirely sure but I think I may only need the "pooler_output" [bs, 768] and not the "last_hidden_state" [bs, 512, 768]
        encoder_output = self.encoder(input_ids=encoder_input["input_ids"], attention_mask=encoder_input["attention_mask"])["pooler_output"]

        # we save some memory by not computing gradients
        # we don't need the computation graph of the code because we won't backprop through the momentum encoder
        with torch.no_grad():
            if isInference:  # use the encoder
                    momentum_encoder_output = self.encoder(input_ids=momentum_encoder_input["input_ids"], attention_mask=momentum_encoder_input["attention_mask"])["pooler_output"]
            else:  # use the momentum encoder
                momentum_encoder_output = self.momentum_encoder(input_ids=momentum_encoder_input["input_ids"], attention_mask=momentum_encoder_input["attention_mask"])["pooler_output"]

        return encoder_output, momentum_encoder_output

    def mlp_forward(self, encoder_mlp_input, positive_momentum_encoder_mlp_input, isInference=False):

        assert(not self.args.disable_mlp)

        encoder_mlp_output = self.encoder_mlp(encoder_mlp_input)
        positive_mlp_output = self.momentum_encoder_mlp(positive_momentum_encoder_mlp_input)
        # only compute the mlp forwards of the queue entries if we're in training
        if not isInference:
            momentum_encoder_mlp_output = self.momentum_encoder_mlp(self.queue) # this is not optimal, because every gpu has to compute the mlp_forward of every queue entry (could in theory split this up and then all_gather, but that's again annoyingly complicated to debug)
            return encoder_mlp_output, positive_mlp_output, momentum_encoder_mlp_output
        else:  # isInference=True
            return encoder_mlp_output, positive_mlp_output

    def training_step(self, batch, batch_idx):
        doc_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}

        current_batch_size = doc_samples["input_ids"].shape[0]

        # print(doc_samples["input_ids"].shape)
        # print(f"In training_step on {self.global_rank}, doc_samples[\"input_ids\"] has shape {doc_samples['input_ids'].shape}")

        # [UPDATE MOMENTUM ENCODER]
        self.update_momentum_encoder()

        # [FORWARD PASS]

        # compute outputs of finetuned CodeBERT encoder and momentum encoder
        encoder_embeddings, positive_momentum_encoder_embeddings = self(doc_samples, code_samples)

        if self.args.normalize_encoder_embeddings_during_training and (not self.args.disable_mlp):  # if the mlp is disabled, we would normalize twice, so we can skip this block
            encoder_embeddings = F.normalize(encoder_embeddings, p=2, dim=1)
            positive_momentum_encoder_embeddings = F.normalize(positive_momentum_encoder_embeddings, p=2, dim=1)

        if not self.args.disable_mlp:
            # encoder_mlp contains the mlp output of the queries
            # pos_mlp_emb contains the mlp output of the positive keys
            # neg_mlp_emb contains the mlp output of all of the negative keys in the queue
            encoder_mlp, pos_mlp_emb, neg_mlp_emb = self.mlp_forward(encoder_embeddings, positive_momentum_encoder_embeddings)

        else:
            encoder_mlp = encoder_embeddings
            pos_mlp_emb = positive_momentum_encoder_embeddings
            neg_mlp_emb = self.concatAllQueueEntries()

        # normalize the length of the embeddings (we want them to be unit vectors for cosine similarity to work correctly)
        encoder_mlp = F.normalize(encoder_mlp, p=2, dim=1)
        if neg_mlp_emb.shape[0] != 0:  # only normalize if non-empty, otherwise normalize() will throw an error
            neg_mlp_emb = F.normalize(neg_mlp_emb, p=2, dim=1)
        pos_mlp_emb = F.normalize(pos_mlp_emb, p=2, dim=1)

        # [UPDATE THE QUEUE]
        self.replaceOldestQueueEntry(positive_momentum_encoder_embeddings)

        # [COMPUTE LOSS]

        # compute similarity of positive NL/PL pairs
        l_pos = torch.bmm(encoder_mlp.view((current_batch_size, 1, -1)), pos_mlp_emb.view((current_batch_size, -1, 1)))

        if neg_mlp_emb.shape[0] != 0:
            # compute similarity of negative NL/PL pairs and concatenate with l_pos to get logits
            l_neg = torch.matmul(encoder_mlp.view((current_batch_size, -1)), torch.transpose(neg_mlp_emb, 0, 1))
            logits = torch.cat((l_pos.view((current_batch_size, 1)), l_neg), dim=1)
        else:
            logits = l_pos.view((current_batch_size, 1))

        # labels: l_pos should always contain the smallest values
        labels = torch.tensor([0 for h in range(current_batch_size)]).type_as(code_samples["input_ids"])  # ugly but does the job

        loss = nn.CrossEntropyLoss()(logits / self.args.temperature, labels)

        self.log("Loss/training", loss.item(), sync_dist=True)

        return loss

    def validation_computations(self, docs_list, code_list, labels, base_path_acc, base_path_sim, substring=""):

        # print(f"validation_computations called on {self.global_rank} with shapes: doc_list - {docs_list.shape}, code_list - {code_list.shape}, labels - {labels.shape} and value labels - {labels}")

        # [COMPARE EVERY QUERY WITH EVERY KEY] (expensive, but necessary for full-corpus accuracy estimation; usually you'd only have one query)

        # [COMPUTE PAIRWISE COSINE SIMILARITY MATRIX]
        logits = torch.matmul(docs_list, torch.transpose(code_list, 0, 1))  # warning: size grows quadratically in the number of validation samples (4 GB at 20k samples)

        # print(f"logits in validation_computations on {self.global_rank} has shape {logits.shape}")

        selection = torch.argmax(logits, dim=1)

        # [COMPUTE TOP1 ACCURACY]
        # the correct guess is always on the diagonal of the logits matrix
        # diagonal_label_tensor = torch.tensor([x for x in range(docs_list.shape[0])]).to(device)

        top_1_correct_guesses = torch.sum(selection == labels)

        top_1_accuracy = top_1_correct_guesses / docs_list.shape[0]  # accuracy is the fraction of correct guesses

        self.log_dict({f"{base_path_acc}/top_1": top_1_accuracy * 100, "step": self.current_epoch}, on_epoch=True, sync_dist=True)

        # [COMPUTE MEAN RECIPROCAL RANK] (MRR)
        # find rank of positive element if the list were sorted (i.e. find number of elements with higher similarity)
        label_list = [logits[i][int(labels[i].item())].item() for i in range(labels.shape[0])]
        label_similarities = torch.tensor(label_list).type_as(docs_list)

        # need to enforce column-wise broadcasting
        ranks = torch.sum(logits >= torch.transpose(label_similarities.view(1, -1), 0, 1),
                          dim=1)  # sum up elements with >= similarity than positive embedding
        mrr = (1 / ranks.shape[0]) * torch.sum(1 / ranks)

        self.log_dict({f"{base_path_acc}/MRR": mrr, "step": self.current_epoch}, on_epoch=True, sync_dist=True)
        if base_path_acc=="Accuracy_enc/validation":
            self.log("hp_metric", mrr, on_epoch=True, sync_dist=True)

        # [COMPUTE TOP5 AND TOP10 ACCURACY]
        # we can reuse the computation for the MRR
        top_5_correct_guesses = torch.sum(ranks <= 5)
        top_10_correct_guesses = torch.sum(ranks <= 10)

        top_5_accuracy = top_5_correct_guesses / docs_list.shape[0]
        top_10_accuracy = top_10_correct_guesses / docs_list.shape[0]
        self.log_dict({f"{base_path_acc}/top_5": top_5_accuracy * 100, "step": self.current_epoch},on_epoch=True, sync_dist=True)
        self.log_dict({f"{base_path_acc}/top_10": top_10_accuracy * 100, "step": self.current_epoch}, on_epoch=True, sync_dist=True)

        # [COMPUTE AVERAGE POSITIVE/NEGATIVE COSINE SIMILARITY]
        avg_pos_cos_similarity = torch.mean(label_similarities)
        self.log_dict({f"{base_path_sim}/cosine/positive": avg_pos_cos_similarity, "step": self.current_epoch}, on_epoch=True, sync_dist=True)

        # sum up all rows, subtract the similarity to the positive sample, then divide by number of samples-1 and finally compute mean over all samples
        avg_neg_cos_similarity = torch.mean((torch.sum(logits, dim=1) - label_similarities) / (code_list.shape[0] - 1))
        self.log_dict({f"{base_path_sim}/cosine/negative": avg_neg_cos_similarity, "step": self.current_epoch}, on_epoch=True, sync_dist=True)

        # free (potentially) a lot of memory
        del label_similarities
        del logits

        # [COMPUTE AVERAGE POSITIVE/NEGATIVE L2 DISTANCE]
        # this might not work...
        l2_distance_matrix = torch.cdist(docs_list, code_list, p=2)  # input: [val_set_size, 768], [val_set_size, 768]; output: [val_set_size, val_set_size] pairwise l2 distance # (similarly to logits above, this becomes huge very fast)

        # print(f"l2_distance_matrix in validation_computation on {self.global_rank} has shape {l2_distance_matrix.shape}")

        l2_label_list = [l2_distance_matrix[i][int(labels[i].item())].item() for i in range(labels.shape[0])]
        label_distances = torch.tensor(l2_label_list).type_as(docs_list)

        avg_pos_l2_distance = torch.mean(label_distances)
        self.log_dict({f"{base_path_sim}/l2/positive": avg_pos_l2_distance, "step": self.current_epoch}, on_epoch=True, sync_dist=True)

        # like for cosine similarity, compute average of negative similarities
        avg_neg_l2_distance = torch.mean((torch.sum(l2_distance_matrix, dim=1) - label_distances) / (code_list.shape[0] - 1))
        self.log_dict({f"{base_path_sim}/l2/negative": avg_neg_l2_distance, "step": self.current_epoch}, on_epoch=True, sync_dist=True)

        # for good measure
        del label_distances
        del l2_distance_matrix

        if True:#self.global_rank==0: # only print from gpu 0 to keep some order in the output
            print(f"Validation {substring} top_1 accuracy (on gpu {self.global_rank}): {top_1_accuracy * 100:.3f}%")
            print(f"Validation {substring} MRR (on gpu {self.global_rank}): {mrr:.4f}")
            print(f"Validation {substring} top_5 accuracy (on gpu {self.global_rank}): {top_5_accuracy * 100:.3f}%")
            print(f"Validation {substring} top_10 accuracy (on gpu {self.global_rank}): {top_10_accuracy * 100:.3f}%")
            print(f"Validation {substring} avg_pos_cos_similarity (on gpu {self.global_rank}): {avg_pos_cos_similarity:.6f}")
            print(f"Validation {substring} avg_neg_cos_similarity (on gpu {self.global_rank}): {avg_neg_cos_similarity:.6f}")
            print(f"Validation {substring} avg_pos_l2_distance (on gpu {self.global_rank}): {avg_pos_l2_distance:.6f}")
            print(f"Validation {substring} avg_neg_l2_distance (on gpu {self.global_rank}): {avg_neg_l2_distance:.6f}")


    def validation_step(self, batch, batch_idx):
        doc_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}

        # print(f"In validation_step on {self.global_rank}, doc_samples[\"input_ids\"] has shape {doc_samples['input_ids'].shape}, code_samples[\"input_ids\"] has shape {code_samples['input_ids'].shape}")

        with torch.no_grad():
            docs_embeddings, code_embeddings = self(doc_samples, code_samples, isInference=True)

            # normalize to ensure correct cosine similarity
            docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
            code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

            # should this be done before or after normalizing the embeddings?
            if not args.disable_mlp:
                mlp_docs, mlp_code = self.mlp_forward(docs_embeddings, code_embeddings, isInference=True)
                mlp_docs = F.normalize(mlp_docs, p=2, dim=1)
                mlp_code = F.normalize(mlp_code, p=2, dim=1)
                return {"index": batch["index"].view(-1), "docs_embeddings": docs_embeddings, "code_embeddings": code_embeddings, "mlp_docs": mlp_docs, "mlp_code": mlp_code}

        # else, don't return the mlp embeddings
        return {"index": batch["index"].view(-1), "docs_embeddings": docs_embeddings, "code_embeddings": code_embeddings}

    def validation_step_end(self, batch_parts): # I know this seems unnecessary at first glance, but the original function would only return the very first entry of every tensor in the dictionary. But we need all entries
        return batch_parts

    def validation_epoch_end(self, outputs):

        # print(f"In validation_epoch_end on {self.global_rank}, outputs[0][\"docs_embeddings\"] has shape {outputs[0]['docs_embeddings'].shape}, outputs[0][\"code_embeddings\"] has shape {outputs[0]['code_embeddings'].shape}")
        # print(f"outputs in validation_epoch_end on gpu {self.global_rank}: {outputs}")

# to validate in parallel, we need to first obtain the complete code embedding matrix.
        # afterwards, we can compute the pairwise cosine similarity, but just on a subset of the data

        #outputs = self.concatAllGather(outputs) # won't need this because validation_step_end already all_gathers # update: that is a big, fat, stupid lie and cost me like 2 hours lmao

        code_emb_list = torch.tensor([]).type_as(outputs[0]["docs_embeddings"])
        if not self.args.disable_mlp:
            mlp_code_list = torch.tensor([]).type_as(outputs[0]["docs_embeddings"])

        for output in outputs:
            code_emb_list = torch.cat((code_emb_list, output["code_embeddings"]), dim=0)
            if not self.args.disable_mlp:
                mlp_code_list = torch.cat((mlp_code_list, output["mlp_code"]), dim=0)

        code_emb_list = self.concatAllGather(code_emb_list)
        mlp_code_list = self.concatAllGather(mlp_code_list)
        # generate tensor that contains only the "assigned" code embeddings
        world_size = torch.cuda.device_count()
        local_rank = self.global_rank

        docs_emb_list = torch.tensor([]).type_as(outputs[0]["docs_embeddings"])
        if not self.args.disable_mlp:
            mlp_docs_list = torch.tensor([]).type_as(outputs[0]["docs_embeddings"])

        for index, output in enumerate(outputs):
            docs_emb_list = torch.cat((docs_emb_list, output["docs_embeddings"]), dim=0)
            if not self.args.disable_mlp:
                mlp_docs_list = torch.cat((mlp_docs_list, output["mlp_docs"]), dim=0)

        basis_index = docs_emb_list.shape[0]
        labels = torch.tensor([x for x in range(basis_index*local_rank, basis_index*(local_rank+1))]).type_as(outputs[0]["docs_embeddings"])
        
        # print(f"in validation_epoch_end on {self.global_rank} with dataset doc shape of {outputs[0]['total_size']}")


        # assert (docs_emb_list.shape == code_emb_list.shape)
        assert (docs_emb_list.shape[1] == 768)  # make sure we use the correct embeddings
        self.validation_computations(docs_emb_list, code_emb_list, labels, "Accuracy_enc/validation", "Similarity_enc", substring="ENC")

        if not self.args.disable_mlp:
            # assert (mlp_docs_list.shape == mlp_code_list.shape)
            assert (mlp_docs_list.shape[1] == 128)  # MLP embeddings are 128-dimensional
            self.validation_computations(mlp_docs_list, mlp_code_list, labels, "Accuracy_MLP/validation", "Similarity_MLP", substring="MLP")

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

    model = MoCoModelPTL(args)

    now = datetime.now()
    now_str = now.strftime("%b%d_%H_%M_%S")
    logger = pl.loggers.TensorBoardLogger("runs", name=f"{now_str}-batch_size_{args.batch_size}-learning_rate_{args.learning_rate}-queue_size_{args.max_queue_size}-max_epochs_{args.num_epochs}-augment_{args.augment}-debug_data_skip_interval_{args.debug_data_skip_interval}-always_use_full_val_{args.always_use_full_val}-disable_mlp_{args.disable_mlp}-num_gpus_{torch.cuda.device_count()}")

    trainer = pl.Trainer(gpus=-1, max_epochs=args.num_epochs, logger=logger, log_every_n_steps=10, flush_logs_every_n_steps=50, reload_dataloaders_every_n_epochs=1, accelerator=args.accelerator, plugins=args.plugins, precision=args.precision)

    trainer.fit(model)

if __name__ == "__main__":
    # [PARSE ARGUMENTS] (if they are given, otherwise keep default value)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_queue_size", type=int, default=64)
    parser.add_argument("--momentum_update_weight", type=float, default=0.999)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--normalize_encoder_embeddings_during_training", type=bool, default=True) #always on for now
    parser.add_argument("--disable_mlp", action="store_true", default=False)
    parser.add_argument("--base_data_folder", type=str, default="datasets/CodeSearchNet")
    parser.add_argument("--debug_data_skip_interval", type=int, default=400) # skips data during the loading process, which effectively makes us use a subset of the original data
    parser.add_argument("--always_use_full_val", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="dp")
    parser.add_argument("--plugins", type=str, default=None)
    parser.add_argument("--precision", type=int, default=16)
    args = parser.parse_args()

    print(f"[HYPERPARAMETERS] Hyperparameters: num_epochs={args.num_epochs}; batch_size={args.batch_size}; learning_rate={args.learning_rate}; temperature={args.temperature}; queue_size={args.max_queue_size}; momentum_update_weight={args.momentum_update_weight}; shuffle={args.shuffle}; augment={args.augment}; DEBUG_data_skip_interval={args.debug_data_skip_interval}; always_use_full_val={args.always_use_full_val}; base_data_folder={args.base_data_folder}; disable_mlp={args.disable_mlp}; seed={args.seed}; num_workers={args.num_workers}, accelerator={args.accelerator}, plugins={args.plugins}")

    execute(args)

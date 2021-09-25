import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from transformers import AutoModel, AutoTokenizer
from pytorch_lightning import LightningModule, Trainer, seed_everything
import pytorch_lightning as pl
import argparse
import random
from datetime import datetime
import numpy as np

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from math import floor, ceil

from utils.multimodal_data_loading import generateDataLoader
from utils.metric_computation import validation_computations
from utils.codeSearchNetDataset import CodeSearchNetDataset

class xMoCoModelPTL(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.docs_tokenizer = None
        self.code_tokenizer = None

        self.docs_fast_encoder = AutoModel.from_pretrained(args.docs_encoder)
        self.docs_slow_encoder = AutoModel.from_pretrained(args.docs_encoder)
        self.code_fast_encoder = AutoModel.from_pretrained(args.code_encoder)
        self.code_slow_encoder = AutoModel.from_pretrained(args.code_encoder)

        self.use_barlow_loss=args.use_barlow_loss
        if args.use_barlow_loss:
            pd = args.barlow_projector_dimension
            if pd == 0:
                self.docs_projector = nn.Identity()
                self.code_projector=nn.Identity()
            else:
                self.docs_projector=nn.Sequential(
                    nn.Linear(768, pd),
                    nn.BatchNorm1d(pd),
                    nn.ReLU(),
                    nn.Linear(pd, pd),
                    nn.BatchNorm1d(pd),
                    nn.ReLU(),
                    nn.Linear(pd, pd)
                )
                if args.barlow_tied_projectors:
                    self.code_projector=self.docs_projector
                else:
                    self.code_projector = nn.Sequential(
                        nn.Linear(768, pd),
                        nn.BatchNorm1d(pd),
                        nn.ReLU(),
                        nn.Linear(pd, pd),
                        nn.BatchNorm1d(pd),
                        nn.ReLU(),
                        nn.Linear(pd, pd)
                    )
            self.barlow_lambda = args.barlow_lambda
            self.barlow_weight = args.barlow_weight
            #self.barlow_batchnorm = nn.BatchNorm1d(768 if pd==0 else pd, affine=False)

        self.effective_queue_size = args.effective_queue_size
        self.update_weight = args.momentum_update_weight
        self.batch_size = args.effective_batch_size
        self.temperature = args.temperature
        self.num_gpus = args.num_gpus
        self.num_hard_negatives = self.args.num_hard_negatives
        self.augment=args.augment

        self.train_loader=None
        self.val_loader=None
        self.raw_data=None

        if args.num_hard_negatives>0:#args.use_hard_negatives:
            #stuff needed for FAISS indexing
            self.negative_docs_queue = None
            self.negative_code_queue = None

            self.hard_negative_queue_size=args.hard_negative_queue_size

            if args.hard_negative_queue_size>0:

                num_hard_negatives_per_iteration = self.batch_size*self.num_hard_negatives
                assert args.hard_negative_queue_size%num_hard_negatives_per_iteration==0, "Assertion error: Hard negative queue size {args.hard_negative_queue_size} is not multiple of {self.batch_size}*{self.num_hard_negatives}={num_hard_negatives_per_iteration}" #make sure that we can't "overshoot" the shape of the queues 

                # queues
                self.register_buffer("hard_negative_docs_queue", torch.randn(self.hard_negative_queue_size, 768))
                self.register_buffer("hard_negative_code_queue", torch.randn(self.hard_negative_queue_size, 768))
                self.hard_negative_docs_queue = F.normalize(self.hard_negative_docs_queue, p=2, dim=1)
                self.hard_negative_code_queue = F.normalize(self.hard_negative_code_queue, p=2, dim=1)

                # index queues
                self.register_buffer("hard_negative_docs_indices", torch.empty(self.hard_negative_queue_size).fill_(-1))
                self.register_buffer("hard_negative_code_indices", torch.empty(self.hard_negative_queue_size).fill_(-1))

                # queue index pointers
                self.register_buffer("hard_negative_docs_current_index", torch.zeros(1, dtype=torch.long)) # the value in those will be identical, but we need both because of how I defined the replaceOldestQueueEntry function
                self.register_buffer("hard_negative_code_current_index", torch.zeros(1, dtype=torch.long))

        if self.effective_queue_size>0: # note: using the combination of effective_queue size=0 and num_hard negatives>0 will break the code!
            # queues
            self.register_buffer("docs_queue", torch.randn(self.effective_queue_size, 768))
            self.register_buffer("code_queue", torch.randn(self.effective_queue_size, 768))

            self.docs_queue = F.normalize(self.docs_queue, p=2, dim=1)
            self.code_queue = F.normalize(self.code_queue, p=2, dim=1)

            # dataset indices of the samples in the queues
            self.register_buffer("docs_indices", torch.empty(self.effective_queue_size).fill_(-1))
            self.register_buffer("code_indices", torch.empty(self.effective_queue_size).fill_(-1))

            # we actually only need one indices and current_index register_buffer, but that would make the replaceOldestQueueEntry implementation a bit nasty. Since these aren't big, I'll just keep it :P
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

        if self.num_hard_negatives>0:
            if self.augment or (self.train_loader==None or self.raw_data==None):
                self.train_loader, self.raw_data = generateDataLoader(self.args.language, "train", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=self.args.shuffle, augment=self.augment, num_workers=self.args.num_workers)
                assert self.raw_data
            generateHardNegativeSearchIndices(self) # need to update the index before every training epoch, regardless of whether we reloaded the data or not
        else:
            if self.augment or self.train_loader==None:
                self.train_loader = generateDataLoader(self.args.language, "train", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=self.args.shuffle, augment=self.augment, num_workers=self.args.num_workers)

        self.num_batches = int(floor(len(self.train_loader)/self.num_gpus))

        return self.train_loader

    @torch.no_grad()
    def val_dataloader(self):
        if self.val_loader==None: # prevent unnecessary validation set reloads by re-using the validation loader
            self.load_tokenizers()
            self.val_loader = generateDataLoader(self.args.language, "valid", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=False, augment=False, num_workers=self.args.num_workers)
        return self.val_loader

    @torch.no_grad()
    def test_dataloader(self):
        self.load_tokenizers()
        # hacky workaround for concatenating two datasets while ensuring that they have the right indices (pytorch's ConcatDataset starts the index at 0 in the second dataset)
        _, raw_data1 = generateDataLoader(self.args.language, "valid", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=False, augment=False, num_workers=self.args.num_workers, return_raw=True)
        _, raw_data2 = generateDataLoader(self.args.language, "test", self.docs_tokenizer, self.code_tokenizer, self.args, shuffle=False, augment=False, num_workers=self.args.num_workers, return_raw=True)

        docs_input_ids = torch.cat((raw_data1.doc_tokens["input_ids"], raw_data2.doc_tokens["input_ids"]), dim=0)
        docs_attention_mask = torch.cat((raw_data1.doc_tokens["attention_mask"], raw_data2.doc_tokens["attention_mask"]), dim=0)
        code_input_ids = torch.cat((raw_data1.code_tokens["input_ids"], raw_data2.code_tokens["input_ids"]), dim=0)
        code_attention_mask = torch.cat((raw_data1.code_tokens["attention_mask"], raw_data2.code_tokens["attention_mask"]), dim=0)

        concat_docs = {"input_ids": docs_input_ids, "attention_mask": docs_attention_mask}
        concat_code = {"input_ids": code_input_ids, "attention_mask": code_attention_mask}

        concat_dataset = CodeSearchNetDataset(concat_docs, concat_code)

        test_dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=int(self.args.effective_batch_size/self.args.num_gpus), drop_last=True)

        return test_dataloader

    # ADAPTED FROM https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/moco/moco2_module.py#L137
    @torch.no_grad()
    def updateMomentumEncoder(self, fast_encoder, slow_encoder):
        """Momentum update of the key encoder."""
        for param_fast, param_slow in zip(fast_encoder.parameters(), slow_encoder.parameters()):
            em = self.update_weight
            param_slow.data = param_slow.data * em + param_fast.data * (1.0 - em)

    @torch.no_grad()
    def replaceOldestQueueEntry(self, queue, queue_indices, current_index, newEntry, newIndices, queue_max_size):
        # gathers the new queue entries from all GPUs and then puts those into the local queue
        gatheredNewEntries = self.concatAllGather(newEntry)
        gatheredNewIndices = self.concatAllGather(newIndices)

        megabatch_size = gatheredNewEntries.shape[0]
        pointer = int(current_index)
        queue[pointer : pointer+megabatch_size, :] = gatheredNewEntries.detach() # the queue doesn't need the computational graph
        queue_indices[pointer : pointer+megabatch_size] = gatheredNewIndices
        current_index[0] = (pointer + megabatch_size) % queue_max_size

    @torch.no_grad()
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

    @torch.no_grad()
    def slow_forward(self, docs_samples, code_samples):
        if (not self.args.enable_training_mode_on_slow_encoders) and (self.docs_slow_encoder.training or self.code_slow_encoder.training):
            self.docs_slow_encoder.eval() # permanently putting the slow encoders into evaluation mode drastically improves the score. it is important to only call .eval() if the model isn't already in evaluation mode, as this is is a fairly expensive operation when called several times per iteration
            self.code_slow_encoder.eval()

        with torch.no_grad():
            slow_docs_embeddings = self.docs_slow_encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
            slow_code_embeddings = self.code_slow_encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
        return slow_docs_embeddings, slow_code_embeddings

    # copied from https://github.com/facebookresearch/barlowtwins/blob/a655214c76c97d0150277b85d16e69328ea52fd9/main.py#L180
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    # adapted from Barlow paper pseudocode
    def barlow_matrix(self, docs_representations, code_representations):
        # projector network forward
        docs_embeddings = self.docs_projector(docs_representations)
        code_embeddings = self.code_projector(code_representations)

        # barlow loss computation
        docs_norm = (docs_embeddings - torch.mean(docs_embeddings, dim=0)) / torch.std(docs_embeddings, dim=0)
        code_norm = (code_embeddings - torch.mean(code_embeddings, dim=0)) / torch.std(code_embeddings, dim=0)

        c = torch.matmul(docs_norm.T, code_norm)

        return c

    def barlow_computations(self, docs_representations, code_representations):

        #topleft = torch.clone(c[0][0])
        #other = torch.clone(c[0][1])

        #topleft = (topleft-1)**2
        #other = (other**2)

        # forward through projector and compute cross-correlation matrix (adapted from Barlow Paper pseudocode)
        c = self.barlow_matrix(docs_representations, code_representations)

        D=c.shape[0]
        c_diff = (c-torch.eye(D, D).type_as(c)).pow(2)

        # adapted from https://discuss.pytorch.org/t/fill-diagonal-of-matrix-with-zero/35083/6
        diagonal_matrix = torch.diag(torch.diag(c_diff)) #remove all off-diagonal elements
        mask = torch.eye(D, D).type_as(c).bool()
        c_diff = c_diff.masked_fill(mask, 0)*self.barlow_lambda+diagonal_matrix

        loss = c_diff.sum()

        self.log("Loss/barlow", loss.item(), sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):

        # update FAISS index in-epoch
        #if self.args.num_hard_negatives>0 and batch_idx % int(ceil(self.num_batches/3)+1) == 0 and batch_idx!=0:
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

        docs_embeddings_unnormalized = docs_embeddings
        code_embeddings_unnormalized = code_embeddings

        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)
        positive_slow_docs_embeddings = F.normalize(positive_slow_docs_embeddings, p=2, dim=1)
        positive_slow_code_embeddings = F.normalize(positive_slow_code_embeddings, p=2, dim=1)

        l_pos_nl_pl = torch.matmul(docs_embeddings, positive_slow_code_embeddings.T)
        l_pos_pl_nl = torch.matmul(code_embeddings, positive_slow_docs_embeddings.T)

        if self.effective_queue_size>0:
            l_neg_nl_pl = torch.matmul(docs_embeddings, self.code_queue.T)
            l_neg_pl_nl = torch.matmul(code_embeddings, self.docs_queue.T)

            logits_nl_pl = torch.cat((l_pos_nl_pl, l_neg_nl_pl), dim=1)
            logits_pl_nl = torch.cat((l_pos_pl_nl, l_neg_pl_nl), dim=1)
        else:
            logits_nl_pl = l_pos_nl_pl
            logits_pl_nl = l_pos_pl_nl

        # mask out false negatives in the regular queues
        if self.args.remove_duplicates:
            for i in range(current_batch_size):
                positive_index=batch_indices[i]
                for j in range(self.effective_queue_size):
                    if positive_index == self.docs_indices[j]: # indices in the regular queue are always the same
                        logits_nl_pl[i][j]=-1
                        logits_pl_nl[i][j]=-1

        # [FIND HARD NEGATIVES] (if enabled)
        if self.num_hard_negatives>0:
            _, hard_negative_docs_indices = self.code_faiss.search(docs_embeddings.detach().cpu().numpy(), self.num_hard_negatives)
            _, hard_negative_code_indices = self.docs_faiss.search(code_embeddings.detach().cpu().numpy(), self.num_hard_negatives)

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

            #hard_negative_docs_embeddings, hard_negative_code_embeddings = self(hard_negative_docs_samples, hard_negative_code_samples, isInference=True)
            hard_negative_docs_embeddings, hard_negative_code_embeddings = self.slow_forward(hard_negative_docs_samples, hard_negative_code_samples)
            hard_negative_docs_embeddings = F.normalize(hard_negative_docs_embeddings, p=2, dim=1)
            hard_negative_code_embeddings = F.normalize(hard_negative_code_embeddings, p=2, dim=1)

            l_hard_neg_nl_pl = torch.matmul(docs_embeddings, hard_negative_code_embeddings.T)
            l_hard_neg_pl_nl = torch.matmul(code_embeddings, hard_negative_docs_embeddings.T)

            l_hard_neg_nl_pl = torch.squeeze(l_hard_neg_nl_pl)
            l_hard_neg_pl_nl = torch.squeeze(l_hard_neg_pl_nl)

            # mask out false negatives in the hard negative part (we always do that, regardless of the remove_duplicates parameter)
            hard_negative_docs_indices = hard_negative_docs_indices.reshape(-1)
            hard_negative_code_indices = hard_negative_code_indices.reshape(-1)
            for i in range(current_batch_size):
                positive_index=batch_indices[i]
                for j in range(hard_negative_docs_indices.shape[0]):
                    if hard_negative_docs_indices[j] == positive_index: #indices in the two hard negative logits might be different
                        l_hard_neg_nl_pl[i][j] = -1
                    if hard_negative_code_indices[j] == positive_index:
                        l_hard_neg_pl_nl[i][j] = -1

            if self.hard_negative_queue_size > 0:
                # compute the similarity on prevîous hard negatives (the ones held in the hard negative queue)
                l_hard_neg_nl_pl_queue = torch.matmul(docs_embeddings, self.hard_negative_code_queue.T)
                l_hard_neg_pl_nl_queue = torch.matmul(code_embeddings, self.hard_negative_docs_queue.T)

                # mask out false negatives in the hard negative queue:
                if self.args.remove_duplicates:
                    for i in range(current_batch_size):
                        positive_index=batch_indices[i]
                        for j in range(self.args.hard_negative_queue_size):
                            if self.hard_negative_docs_indices[j] == positive_index: # indices in the two hard negative queues might be different
                                l_hard_neg_nl_pl_queue[i][j] = -1
                            if self.hard_negative_code_indices[j] == positive_index:
                                l_hard_neg_pl_nl_queue[i][j] = -1

                # concat results with in-batch logits
                logits_nl_pl = torch.cat((logits_nl_pl, l_hard_neg_nl_pl_queue), dim=1)
                logits_pl_nl = torch.cat((logits_pl_nl, l_hard_neg_pl_nl_queue), dim=1)

                # update the hard negative queues
                self.replaceOldestQueueEntry(self.hard_negative_docs_queue, self.hard_negative_docs_indices, self.hard_negative_docs_current_index, hard_negative_docs_embeddings, torch.from_numpy(hard_negative_docs_indices).view(-1).type_as(self.hard_negative_docs_current_index), self.hard_negative_queue_size)
                self.replaceOldestQueueEntry(self.hard_negative_code_queue, self.hard_negative_code_indices, self.hard_negative_code_current_index, hard_negative_code_embeddings, torch.from_numpy(hard_negative_code_indices).view(-1).type_as(self.hard_negative_code_current_index), self.hard_negative_queue_size)

            logits_nl_pl = torch.cat((logits_nl_pl, l_hard_neg_nl_pl), dim=1)
            logits_pl_nl = torch.cat((logits_pl_nl, l_hard_neg_pl_nl), dim=1)

        labels = torch.tensor(range(logits_nl_pl.shape[0])).type_as(code_samples["input_ids"])

        loss_nl_pl = nn.CrossEntropyLoss()(logits_nl_pl/self.temperature, labels)

        loss_pl_nl = nn.CrossEntropyLoss()(logits_pl_nl/self.temperature, labels)

        loss_contrast = (loss_nl_pl + loss_pl_nl) / 2.0

        loss = loss_contrast

        if self.use_barlow_loss:
            #loss_barlow = self.barlow_computations(docs_embeddings_unnormalized, code_embeddings_unnormalized) # seems it makes no difference whether we use this or the normalized ones, but I'll now stick to this as it is closer to what the Barlow paper did
            loss_barlow = self.barlow_computations(docs_embeddings, code_embeddings)
            #loss = loss_contrast + self.args.barlow_weight*loss_barlow # note that this is not strictly a "tradeoff" in the classical sense "(1-x) * a + x * b" but rather an additional loss. this seems to produce much better results
            loss = (1-self.args.barlow_weight) * loss_contrast + self.args.barlow_weight * loss_barlow

        # [LOGGING]
        self.log("Loss/contrast", loss_contrast.item(), sync_dist=True)
        self.log("Loss/combined", loss.item(), sync_dist=True)

        # [UPDATE THE QUEUES]
        if self.effective_queue_size>0: # the case queue_size=0 isn't handled anywhere else
            self.replaceOldestQueueEntry(self.docs_queue, self.docs_indices, self.docs_current_index, positive_slow_docs_embeddings, batch_indices, self.effective_queue_size)
            self.replaceOldestQueueEntry(self.code_queue, self.code_indices, self.code_current_index, positive_slow_code_embeddings, batch_indices, self.effective_queue_size)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        docs_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}

        docs_embeddings, code_embeddings = self(docs_samples, code_samples, isInference=True)

        if batch_idx == 0:
            matrix_grid = torch.unsqueeze(self.barlow_matrix(docs_embeddings, code_embeddings), 0)#make_grid(self.barlow_matrix(docs_embeddings, code_embeddings), normalize=True)
            matrix_grid.div_(int(self.args.effective_batch_size/self.args.num_gpus))
            tb = self.logger.experiment
            tb.add_image(f"CC matrix in epoch {self.current_epoch}", matrix_grid)

        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        return {"index": batch["index"].view(-1), "docs_embeddings": docs_embeddings, "code_embeddings": code_embeddings}

    @torch.no_grad()
    def validation_step_end(self, batch_parts): # I know this seems unnecessary at first glance, but the original function would only return the very first entry of every tensor in the dictionary. But we need all entries
        return batch_parts

    @torch.no_grad()
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

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if  self.docs_fast_encoder.training or self.code_fast_encoder.training: #apparently PTL doesn't do that automatically when testing
            self.docs_fast_encoder.eval()
            self.code_fast_encoder.eval()

        docs_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}

        docs_embeddings, code_embeddings = self(docs_samples, code_samples, isInference=True)

        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        return {"index": batch["index"].view(-1), "docs_embeddings": docs_embeddings, "code_embeddings": code_embeddings}

    @torch.no_grad()
    def test_step_end(self, batch_parts):
        return batch_parts

    @torch.no_grad()
    def test_epoch_end(self, outputs):
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
        validation_computations(self, docs_emb_list, code_emb_list, labels, "Accuracy_enc/TEST", "Similarity_TEST", substring="ENC")

def execute(args):

    if args.num_hard_negatives>0:
        import os
        os.environ["TOKENIZERS_PARALLELISM"]="false"
        os.environ["OMP_WAIT_POLICY"]="PASSIVE"
        global generateHardNegativeSearchIndices
        from utils.hard_negative_search import generateHardNegativeSearchIndices

    # set all seeds so we can ensure same MLP initialization and augmentation behaviour on all GPUs
    seed_everything(args.seed, workers=False)


    now = datetime.now()
    now_str = now.strftime("%b%d_%H_%M_%S")
    logger = TensorBoardLogger("runs", name=f"{now_str}-xMoCo-language_{args.language}-eff_bs_{args.effective_batch_size}-lr_{args.learning_rate}-eff_qs_{args.effective_queue_size}-max_epochs_{args.num_epochs}-aug_{args.augment}-shuf_{args.shuffle}-debug_skip_interval_{args.debug_data_skip_interval}-always_full_val_{args.always_use_full_val}-docs_enc_{args.docs_encoder}-code_enc_{args.code_encoder}-num_gpus_{args.num_gpus}-rmv_dup_{args.remove_duplicates}-use_hard_negatives_{args.num_hard_negatives>0}-num_hard_negatives_{args.num_hard_negatives}-hard_negative_queue_size_{args.hard_negative_queue_size}-tm_on_slow_{args.enable_training_mode_on_slow_encoders}-use_barlow_{args.use_barlow_loss}-barl_pd_{args.barlow_projector_dimension}-barl_lambd_{args.barlow_lambda}-barl_weight_{args.barlow_weight}")

    if not args.skip_training:
        model = xMoCoModelPTL(args)

        early_stopping_callback = EarlyStopping(monitor="Accuracy_enc/validation/MRR", patience=3, mode="max")
        if args.generate_checkpoints:
            checkpoint_callback = ModelCheckpoint(monitor="Accuracy_enc/validation/MRR", dirpath = "/itet-stor/jstuder/net_scratch/nl-pl_moco/checkpoints/", filename=(str(now_str)+"-xMoCo-"+str(args.language)), mode="max")
            callbacks=[checkpoint_callback, early_stopping_callback]
        else:
            callbacks=[early_stopping_callback]

        trainer = Trainer(callbacks=callbacks, val_check_interval=1.0, gpus=args.num_gpus, max_epochs=args.num_epochs, logger=logger, log_every_n_steps=10, flush_logs_every_n_steps=50, reload_dataloaders_every_n_epochs=(1 if (args.augment or args.num_hard_negatives>0) else 0), accelerator=args.accelerator, plugins=args.plugins, precision=args.precision)

        trainer.fit(model)

        try:
            print(f"Training done. Best checkpoint: {checkpoint_callback.best_model_path} with validation MRR {checkpoint_callback.best_model_score}")
        except:
            print("No checkpoint found.")

    if args.do_test: #note: currently, this will crash the program because the folder somehow doesn't get generated; will need to fix this later
        # load best checkpoint from training
        if not args.skip_training and args.checkpoint_path!=None:
            model = xMoCoModelPTL.load_from_checkpoint(checkpoint_callback.best_model_path) # this somehow doesn't work, we always need to run the program again with --skip_training and --do_test and the checkpoint that was generated during training
        else:
            trainer = Trainer(gpus=args.num_gpus, logger=logger, accelerator=args.accelerator, plugins=args.plugins, precision=args.precision)
            model = xMoCoModelPTL.load_from_checkpoint(args.checkpoint_path)

        # run the actual tests (data is loaded in model.test_dataloader())
        trainer.test(model=model)

if __name__ == "__main__":
    # [PARSE ARGUMENTS] (if they are given, otherwise keep default value)
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_encoder", type=str, default="microsoft/codebert-base")
    parser.add_argument("--code_encoder", type=str, default="microsoft/codebert-base")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--effective_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--effective_queue_size", type=int, default=4096)
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
    parser.add_argument("--num_hard_negatives", type=int, default=0)
    parser.add_argument("--hard_negative_queue_size", type=int, default=0)
    parser.add_argument("--enable_training_mode_on_slow_encoders", action="store_true", default=False)
    parser.add_argument("--use_barlow_loss", action="store_true", default=False)
    parser.add_argument("--barlow_projector_dimension", type=int, default=0)
    parser.add_argument("--barlow_lambda", type=float, default=0.005)
    parser.add_argument("--barlow_weight", type=float, default=5e-5)
    parser.add_argument("--barlow_tied_projectors", action="store_true", default=False)
    parser.add_argument("--skip_training", action="store_true", default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--do_test", action="store_true", default=False)
    parser.add_argument("--generate_checkpoints", action="store_true", default=False)
    args = parser.parse_args()

    print(f"[HYPERPARAMETERS] Hyperparameters: xMoCo - language={args.language} - num_epochs={args.num_epochs}; effective_batch_size={args.effective_batch_size}; learning_rate={args.learning_rate}; temperature={args.temperature}; effective_queue_size={args.effective_queue_size}; momentum_update_weight={args.momentum_update_weight}; shuffle={args.shuffle}; augment={args.augment}; DEBUG_data_skip_interval={args.debug_data_skip_interval}; always_use_full_val={args.always_use_full_val}; base_data_folder={args.base_data_folder}; seed={args.seed}; num_workers={args.num_workers}, accelerator={args.accelerator}, plugins={args.plugins}, num_gpus={args.num_gpus}, remove_duplicates={args.remove_duplicates}, language={args.language}, use_hard_negatives={args.num_hard_negatives>0}, num_hard_negatives={args.num_hard_negatives}; hard_negative_queue_size={args.hard_negative_queue_size}; enable_training_mode_on_slow_encoders={args.enable_training_mode_on_slow_encoders}, use_barlow_loss={args.use_barlow_loss}, barlow_projector_dimension={args.barlow_projector_dimension}, barlow_lambda={args.barlow_lambda}, barlow_weight={args.barlow_weight}")

    execute(args)


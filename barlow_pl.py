import torch
from torch import nn
import torch.nn.functional as F
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

from utils.improved_data_loading import generateDataLoader
from utils.metric_computation import validation_computations

class BarlowTwinsPTL(LightningModule):
    def __init__(self, args, tuneDict=None):
        super().__init__()

        if tuneDict!=None:
            pass # replace arguments in args for hyperparameter tuning

        self.args=args

        self.tokenizer = None # can't init this here (before the multiprocessing fork) because then huggingface will be unhappy

        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.projector = nn.Sequential(
            nn.Linear(768, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192)
        )

        self.batch_size = args.effective_batch_size
        self.num_gpus = args.num_gpus
        self.learning_rate = args.learning_rate
        self.lambd = args.lambd

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def load_tokenizer(self):
        if self.tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

    def train_dataloader(self):
        self.load_tokenizer()
        train_loader = generateDataLoader(self.args.language, "train", self.tokenizer, self.args, batch_size=int(self.batch_size/self.num_gpus), shuffle=self.args.shuffle, augment=self.args.augment, num_workers=self.args.num_workers)
        return train_loader

    def val_dataloader(self):
        self.load_tokenizer()
        val_dataloader = generateDataLoader(self.args.language, "validate", self.tokenizer, self.args, batch_size=int(self.batch_size/self.num_gpus), shuffle=False, augment=False, num_workers=self.args.num_workers)
        return val_dataloader

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
            docs_representations = self.encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
            code_representations = self.encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
            docs_embeddings = self.projector(docs_representations)
            code_embeddings = self.projector(code_representations)
        else:
            with torch.no_grad():
                docs_representations = self.encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
                code_representations = self.encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
                docs_embeddings = self.projector(docs_representations)
                code_embeddings = self.projector(code_representations)
                #return docs_representations, code_representations
        return docs_embeddings, code_embeddings

    def training_step(self, batch, batch_idx):
        docs_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}
        batch_indices = batch["index"]
        current_batch_size = docs_samples["input_ids"].shape[0]

        docs_embeddings, code_embeddings = self(docs_samples, code_samples)

        docs_norm = (docs_embeddings - torch.mean(docs_embeddings, dim=0)) / torch.std(docs_embeddings, dim=0)
        code_norm = (code_embeddings - torch.mean(code_embeddings, dim=0)) / torch.std(code_embeddings, dim=0)

        c = torch.matmul(docs_norm.T, code_norm) # found a simpler way than torch.transpose: just use tensor.T

        D = c.shape[0]
        c_diff = (c-torch.eye(D)).pow(2)

        diagonal_indices = [i*D+i for i in range(D)]
        off_diagonal_indices = [x for x in range(D**2) if x not in diagonal_indices]

        # multiply off-diagonal elements by lambda
        c_diff.view(-1)[off_diagonal_indices]*=self.lambd

        loss = c_diff.sum()

        self.log("Loss/training", loss.item(), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        docs_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}
        batch_indices = batch["index"]
        current_batch_size = docs_samples["input_ids"].shape[0]

        docs_embeddings, code_embeddings = self(docs_samples, code_samples, isInference=True)

        docs_norm = (docs_embeddings - torch.mean(docs_embeddings, dim=0)) / torch.std(docs_embeddings, dim=0)
        code_norm = (code_embeddings - torch.mean(code_embeddings, dim=0)) / torch.std(code_embeddings, dim=0)

        c = torch.matmul(docs_norm.T, code_norm)  # found a simpler way than torch.transpose: just use tensor.T

        D = c.shape[0]
        c_diff = (c - torch.eye(D)).pow(2)

        diagonal_indices = [i * D + i for i in range(D)]
        off_diagonal_indices = [x for x in range(D ** 2) if x not in diagonal_indices]

        # multiply off-diagonal elements by lambda
        c_diff.view(-1)[off_diagonal_indices] *= self.lambd

        loss = c_diff.sum()

        self.log("Loss/validation", loss.item(), sync_dist=True, on_epoch=True) # on_epoch is actually True by default in the validation_stpe, but whatever

def execute(args):

    seed_everything(args.seed, workers=False)

    model = BarlowTwinsPTL(args)

    now = datetime.now()
    now_str = now.strftime("%b%d_%H_%M_%S")
    logger = TensorBoardLogger("runs", name=f"{now_str}-BarlowTwins-language_{args.language}-eff_bs_{args.effective_batch_size}-lr_{args.learning_rate}-lambda_{args.lambd}-max_epochs_{args.num_epochs}-aug_{args.augment}-shuf_{args.shuffle}-debug_skip_interval_{args.debug_data_skip_interval}-always_full_val_{args.always_use_full_val}-encoder_{args.model_name}-num_gpus_{args.num_gpus}")

    if args.do_tune:
        pass # TODO: add tuner config and function
        return

    early_stopping_callback = EarlyStopping(monitor="Loss/validation", patience=3, mode="min")

    checkpoint_callback = ModelCheckpoint(monitor="Loss/validation",
                                          dirpath="/itet-stor/jstuder/net_scratch/nl-pl_moco/checkpoints/",
                                          filename=(str(now_str)+"-BarlowTwins-"+str(args.language)),
                                          mode="min")

    callbacks = [early_stopping_callback, checkpoint_callback]

    trainer = Trainer(callbacks=callbacks, val_check_interval=1.0, gpus=args.num_gpus, max_epochs=args.num_epochs,
                      logger=logger, reload_dataloaders_every_n_epochs=1,
                      accelerator=args.accelerator, plugins=args.plugins, precision=args.precision)

    trainer.fit(model)

    print(f"Training done. Best checkpoint: {checkpoint_callback.best_model_path} with validation MRR {checkpoint_callback.best_model_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lambd", type=float, default=5e-3)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--base_data_folder", type=str, default="/itet-stor/jstuder/net_scratch/nl-pl_moco/datasets/CodeSearchNet")
    parser.add_argument("--debug_data_skip_interval", type=int, default=1)
    parser.add_argument("--always_use_full_val", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="dp")
    parser.add_argument("--plugins", type=str, default=None)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--language", type=str, default="ruby")
    parser.add_argument("--do_tune", action="store_true", default=False)
    parser.add_argument("--do_test", action="store_true", default=False) # doesn't do anything for now
    args = parser.parse_args()

    if args.do_tune:
        print(f"THIS IS A HYPERPARAMETER TUNING RUN!")
    else:
        print(f"[HYPERPARAMETERS] BarlowTwins - "
              f"language={args.language} - effective_batch_size={args.effective_batch_size} - "
              f"learning_rate={args.learning_rate} - lambd={args.lambd} - shuffle={args.shuffle} - augment={args.augment} - "
              f"num_epochs={args.num_epochs} - gpus={args.num_gpus} - debug_data_skip_interval={args.debug_data_skip_interval} - use_full_val={args.always_use_full_val} - "
              f"seed={args.seed} - do_test={args.do_test} - model_name={args.model_name}")

    execute(args)
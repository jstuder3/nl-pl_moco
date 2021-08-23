import faiss
import numpy as np
import torch
import torch.nn.functional as F
from math import floor
import sys

@torch.no_grad()
def generateHardNegativeSearchIndices(self):

    # this function should generate two separate FAISS indices: one for the docs samples and one for the code samples.
    # these indices can then be queried with vectors to obtain the k nearest neighbours in terms of inner product similarity (or in our case cosine similarity because the vectors are normalized)

    assert self.args.use_hard_negatives
    
    # this might be counterproductive... not entirely sure...
    # self.docs_fast_encoder.eval()
    # self.code_fast_encoder.eval()

    local_rank = self.global_rank
    data = self.raw_data
    num_gpus = self.num_gpus
    batch_size = int(self.batch_size/num_gpus) # for inference, we could in theory use a much larger batch size because we don't have to store the computational graph

    print(f"Generating FAISS index on GPU {local_rank}...")

    # generate index
    self.docs_faiss = faiss.IndexFlatIP(768)# output dimension of encoders is generally 768
    self.code_faiss = faiss.IndexFlatIP(768)

    # feed all of the data designated for this GPU through the encoders (for correct indexing, it is important that they stay in the same order as they are in the raw dataset)
    num_iterations = floor(len(data)/(num_gpus*batch_size)) # "cut off" any incomplete batch and manually compute the embeddings for that cut off batch on a single GPU (for simplicity)
    for iteration in range(num_iterations):
        if local_rank==0 and iteration%10==0:
            sys.stdout.write(f"\rGenerating FAISS index: {iteration} / {num_iterations} ({(iteration/num_iterations)*100:.1f}%)")
            sys.stdout.flush()
        start_index = int(iteration*batch_size*num_gpus + local_rank*batch_size)
        batch = data[start_index : start_index+batch_size]
        docs_samples = {"input_ids": batch["doc_input_ids"].type_as(self.docs_current_index), "attention_mask": batch["doc_attention_mask"].type_as(self.docs_current_index)}
        code_samples = {"input_ids": batch["code_input_ids"].type_as(self.docs_current_index), "attention_mask": batch["code_attention_mask"].type_as(self.docs_current_index)}
        docs_embeddings, code_embeddings = self.forward(docs_samples, code_samples, isInference=True)

        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        docs_gathered = self.concatAllGather(docs_embeddings).cpu()
        code_gathered = self.concatAllGather(code_embeddings).cpu()

        self.docs_faiss.add(docs_gathered.numpy()) # warning: this might require the use of float32 even though our computations all use float16
        self.code_faiss.add(code_gathered.numpy())

    # if there's a "last batch" that is not a full batch, compute the embeddings for that on the main gpu
    if num_iterations!=(len(data)/(num_gpus*batch_size)):
        last_docs_embeddings = torch.tensor([]).type_as(self.docs_queue)
        last_code_embeddings = torch.tensor([]).type_as(self.docs_queue)
        if local_rank==0 and num_iterations!=(len(data)/(num_gpus*batch_size)): #compute the final snippet on the rank 0 gpu if we haven't reached the end yet
            start_index = num_iterations*batch_size*num_gpus
            batch = data[start_index : -1]
            docs_samples = {"input_ids": batch["doc_input_ids"].type_as(self.docs_current_index), "attention_mask": batch["doc_attention_mask"].type_as(self.docs_current_index)}
            code_samples = {"input_ids": batch["code_input_ids"].type_as(self.docs_current_index), "attention_mask": batch["code_attention_mask"].type_as(self.docs_current_index)}
            last_docs_embeddings, last_code_embeddings = self.forward(docs_samples, code_samples, isInference=True)
            last_docs_embeddings = F.normalize(last_docs_embeddings, p=2, dim=1)
            last_code_embeddings = F.normalize(last_code_embeddings, p=2, dim=1)
        last_docs_gathered = self.concatAllGather(last_docs_embeddings).cpu()
        last_code_gathered = self.concatAllGather(last_code_embeddings).cpu()

        self.docs_faiss.add(last_docs_gathered.numpy())
        self.code_faiss.add(last_code_gathered.numpy())

    sys.stdout.write(f"\rFAISS index generation completed (added {self.docs_faiss.ntotal} samples to FAISS index)\n")
    sys.stdout.flush()

    print(f"Construction of FAISS index finished on GPU {local_rank}. Docs index contains {self.docs_faiss.ntotal} elements, Code index contains {self.code_faiss.ntotal} elements")

    # this might be counterproductive... not entirely sure...
    # self.docs_fast_encoder.train()
    # self.code_fast_encoder.train()


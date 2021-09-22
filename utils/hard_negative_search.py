import faiss
import numpy as np
import torch
import torch.nn.functional as F
from math import floor
import sys
import time

@torch.no_grad()
def generateHardNegativeSearchIndices(self):

    # this function should generate two separate FAISS indices: one for the docs samples and one for the code samples.
    # these indices can then be queried with vectors to obtain the k nearest neighbours in terms of inner product similarity (or in our case cosine similarity because the vectors are normalized)

    assert self.args.num_hard_negatives>0

    self.negative_docs_queue = torch.tensor([]).type_as(self.hard_negative_docs_queue).float() #yeah, I know this is ugly
    self.negative_code_queue = torch.tensor([]).type_as(self.hard_negative_code_queue)

    local_rank = self.global_rank
    data = self.raw_data
    num_gpus = self.num_gpus
    batch_size = int(self.batch_size/num_gpus) # for inference, we could in theory use a much larger batch size because we don't have to store the computational graph

    print(f"Generating FAISS index on GPU {local_rank}...")

    # feed all of the data designated for this GPU through the encoders (for correct indexing, it is important that they stay in the same order as they are in the raw dataset)
    num_iterations = floor(len(data)/(num_gpus*batch_size)) # "cut off" any incomplete batch and manually compute the embeddings for that cut off batch on a single GPU (for simplicity)
    for iteration in range(num_iterations):
        if local_rank==0 and iteration%10==0:
            sys.stdout.write(f"\rGenerating FAISS index: {iteration} / {num_iterations} ({(iteration/num_iterations)*100:.1f}%)")
            sys.stdout.flush()
        start_index = int(iteration*batch_size*num_gpus + local_rank*batch_size)
        batch = data[start_index : start_index+batch_size]
        docs_samples = {"input_ids": batch["doc_input_ids"].type_as(self.hard_negative_docs_current_index), "attention_mask": batch["doc_attention_mask"].type_as(self.hard_negative_docs_current_index)}
        code_samples = {"input_ids": batch["code_input_ids"].type_as(self.hard_negative_docs_current_index), "attention_mask": batch["code_attention_mask"].type_as(self.hard_negative_docs_current_index)}

        #docs_embeddings, code_embeddings = self.forward(docs_samples, code_samples, isInference=True)
        docs_embeddings, code_embeddings = self.slow_forward(docs_samples, code_samples) # need to check whether this makes any difference over using the slow encoders

        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        docs_gathered = self.concatAllGather(docs_embeddings)
        code_gathered = self.concatAllGather(code_embeddings)

        self.negative_docs_queue = torch.cat((self.negative_docs_queue, docs_gathered), dim=0)
        self.negative_code_queue = torch.cat((self.negative_code_queue, code_gathered), dim=0)
        #self.docs_faiss.add(docs_gathered.numpy()) # warning: this might require the use of float32 even though our computations all use float16
        #self.code_faiss.add(code_gathered.numpy())

    self.docs_faiss = faiss.IndexFlatIP(768)
    self.code_faiss = faiss.IndexFlatIP(768)

    st = time.time()
    self.docs_faiss.add(self.negative_docs_queue.cpu().numpy())
    self.code_faiss.add(self.negative_code_queue.cpu().numpy())
    print(f"Adding to index took {time.time()-st} seconds")
    print(f"finished on gpu {local_rank}")

    sys.stdout.write(f"FAISS index generation completed (added {self.docs_faiss.ntotal} samples to FAISS index)\n")
    sys.stdout.flush()
    #print(f"Full forward pass done on gpu {local_rank}, forwarded {self.negative_docs_queue.shape[0]} samples")
    print(f"Construction of FAISS index finished on GPU {local_rank}. Docs index contains {self.docs_faiss.ntotal} elements, Code index contains {self.code_faiss.ntotal} elements")




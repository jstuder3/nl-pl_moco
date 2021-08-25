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

    assert self.args.use_hard_negatives
    
    self.negative_docs_queue = torch.tensor([]).type_as(self.docs_queue).float() #yeah, I know this is ugly
    self.negative_code_queue = torch.tensor([]).type_as(self.negative_docs_queue)

    # this might be counterproductive... not entirely sure...
    # self.docs_fast_encoder.eval()
    # self.code_fast_encoder.eval()

    local_rank = self.global_rank
    data = self.raw_data
    num_gpus = self.num_gpus
    batch_size = int(self.batch_size/num_gpus) # for inference, we could in theory use a much larger batch size because we don't have to store the computational graph

    print(f"Generating FAISS index on GPU {local_rank}...")

    # generate index
    docs_faiss_cpu = faiss.IndexFlatIP(768)# output dimension of encoders is generally 768
    code_faiss_cpu = faiss.IndexFlatIP(768)

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

        docs_gathered = self.concatAllGather(docs_embeddings)
        code_gathered = self.concatAllGather(code_embeddings)

        self.negative_docs_queue = torch.cat((self.negative_docs_queue, docs_gathered), dim=0)
        self.negative_code_queue = torch.cat((self.negative_code_queue, code_gathered), dim=0)
        #self.docs_faiss.add(docs_gathered.numpy()) # warning: this might require the use of float32 even though our computations all use float16
        #self.code_faiss.add(code_gathered.numpy())

    # if there's a "last batch" that is not a full batch, compute the embeddings for that on the main gpu
    # warning: this contains an error somewher which causes stuff to deadlock on the gpu process that processes the last batch
    #if num_iterations!=(len(data)/(num_gpus*batch_size)):
    #    last_docs_embeddings = torch.tensor([]).type_as(self.docs_queue)
    #    last_code_embeddings = torch.tensor([]).type_as(self.docs_queue)
    #    if local_rank==0: #compute the final snippet on the rank 0 gpu if we haven't reached the end yet
    #        start_index = num_iterations*batch_size*num_gpus
    #        batch = data[start_index : -1]
    #        docs_samples = {"input_ids": batch["doc_input_ids"].type_as(self.docs_current_index), "attention_mask": batch["doc_attention_mask"].type_as(self.docs_current_index)}
    #        code_samples = {"input_ids": batch["code_input_ids"].type_as(self.docs_current_index), "attention_mask": batch["code_attention_mask"].type_as(self.docs_current_index)}
    #        last_docs_embeddings, last_code_embeddings = self.forward(docs_samples, code_samples, isInference=True)
    #        last_docs_embeddings = F.normalize(last_docs_embeddings, p=2, dim=1)
    #        last_code_embeddings = F.normalize(last_code_embeddings, p=2, dim=1)
    #    last_docs_gathered = self.concatAllGather(last_docs_embeddings)
    #    last_code_gathered = self.concatAllGather(last_code_embeddings)

        #self.docs_faiss.add(last_docs_gathered.numpy())
        #self.code_faiss.add(last_code_gathered.numpy())
    #    self.negative_docs_queue = torch.cat((self.negative_docs_queue, last_docs_gathered), dim=0)
    #    self.negative_code_queue = torch.cat((self.negative_code_queue, last_code_gathered), dim=0)

    #print(f"start adding elements to indices on gpu {local_rank}")
    #print(f"gpu {local_rank}: docs_faiss - {self.docs_faiss}")
    #print(f"gpu {local_rank}: code_faiss - {self.code_faiss}")
    #if local_rank == 0:
    #    import IPython
    #    IPython.embed()
    #self.negative_docs_numpy = self.negative_docs_queue.cpu().numpy()
    #self.negative_code_numpy = self.negative_code_queue.cpu().numpy()
    #if local_rank==0:
    #    import IPython
    #    IPython.embed()

    #res = faiss.StandardGpuResources()

    #st = time.time()

    self.docs_faiss = faiss.IndexFlatIP(768)
    self.code_faiss = faiss.IndexFlatIP(768)

    #self.docs_faiss = faiss.IndexIVFFlat(docs_faiss_cpu, 768, 40)
    #self.code_faiss = faiss.IndexIVFFlat(code_faiss_cpu, 768, 40)

    #self.docs_faiss.train(self.negative_docs_queue.cpu().numpy())
    #self.code_faiss.train(self.negative_code_queue.cpu().numpy())

    #print(f"Training took {time.time()-st} seconds")

    #self.docs_faiss = faiss.index_cpu_to_gpu(res, 0, docs_faiss_cpu)
    #self.code_faiss = faiss.index_cpu_to_gpu(res, 0, code_faiss_cpu)

    st = time.time()
    self.docs_faiss.add(self.negative_docs_queue.cpu().numpy())
    self.code_faiss.add(self.negative_code_queue.cpu().numpy())
    print(f"Adding to index took {time.time()-st} seconds")
    print(f"finished on gpu {local_rank}")

    sys.stdout.write(f"FAISS index generation completed (added {self.docs_faiss.ntotal} samples to FAISS index)\n")
    sys.stdout.flush()
    #print(f"Full forward pass done on gpu {local_rank}, forwarded {self.negative_docs_queue.shape[0]} samples")
    print(f"Construction of FAISS index finished on GPU {local_rank}. Docs index contains {self.docs_faiss.ntotal} elements, Code index contains {self.code_faiss.ntotal} elements")

    # this might be counterproductive... not entirely sure...
    # self.docs_fast_encoder.train()
    # self.code_fast_encoder.train()


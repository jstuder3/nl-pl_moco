import faiss
import numpy as np
import torch
import torch.nn.functional as F
from math import floor
import sys
import time

# COPIED FROM https://github.com/facebookresearch/faiss/issues/878
def my_index_cpu_to_gpu_multiple(resources, index, co=None, gpu_nos=None):
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if gpu_nos is None:
        gpu_nos = range(len(resources))
    for i, res in zip(gpu_nos, resources):
        vdev.push_back(i)
        vres.push_back(res)
    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
    index.referenced_objects = resources
    return index


@torch.no_grad()
def generateHardNegativeSearchIndices(self):
    # this function should generate a FAISS index by forwarding all of the code samples through the encoder
    # this index can then be queried with vectors to obtain the k nearest neighbours in terms of inner product similarity (or in our case cosine similarity because the vectors are normalized)

    assert self.args.use_hard_negatives

    self.negative_matrix = torch.tensor([]).cuda().float()  # yeah, I know this is ugly

    local_rank = self.global_rank
    data = self.raw_data
    num_gpus = self.num_gpus
    batch_size = int(self.batch_size / num_gpus)  # for inference, we could in theory use a much larger batch size because we don't have to store the computational graph

    print(f"Generating FAISS index on GPU {local_rank}...")

    # feed all of the data designated for this GPU through the encoders (for correct indexing, it is important that they stay in the same order as they are in the raw dataset)
    num_iterations = floor(len(data) / (num_gpus * batch_size))  # "cut off" any incomplete batch and manually compute the embeddings for that cut off batch on a single GPU (for simplicity)
    for iteration in range(num_iterations):
        if local_rank == 0 and iteration % 10 == 0:
            sys.stdout.write(f"\rGenerating FAISS index: {iteration} / {num_iterations} ({(iteration / num_iterations) * 100:.1f}%)")
            sys.stdout.flush()
        start_index = int(iteration * batch_size * num_gpus + local_rank * batch_size)
        batch = data[start_index: start_index + batch_size]
        code_samples = {"input_ids": batch["code_input_ids"].cuda().long(), "attention_mask": batch["code_attention_mask"].cuda().long()}

        with torch.no_grad():
            code_embeddings = self.encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]

        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        code_gathered = self.concatAllGather(code_embeddings)

        self.negative_matrix = torch.cat((self.negative_matrix, code_gathered), dim=0)

    #faiss_index_cpu = faiss.IndexFlatIP(768)
    self.faiss_index = faiss.IndexFlatIP(768)
    #cloner_options = faiss.GpuClonerOptions()
    #cloner_options.useFloat16=True

    #config = faiss.GpuIndexFlatConfig()
    #config.device=0

    res = faiss.StandardGpuResources()
    res.setLogMemoryAllocations(True)

#    co = faiss.GpuClonerOptions()
#    co.useFloat16 = True

    #self.faiss_index = my_index_cpu_to_gpu_multiple([res], faiss_index_cpu, co=None, gpu_nos=[self.global_rank])
    #self.faiss_index = faiss.GpuIndexFlatIP(res, 768, config)
    #self.faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index_cpu, cloner_options)#faiss.index_cpu_to_all_gpus(faiss_index_cpu)#my_index_cpu_to_gpu_multiple(res, faiss_index_cpu)#faiss.index_cpu_to_gpu(res, self.global_rank, faiss_index_cpu)

    st = time.time()
    self.faiss_index.add(self.negative_matrix.cpu().numpy())
    print(f"Adding to index took {time.time() - st} seconds")
    print(f"finished on gpu {local_rank}")

    sys.stdout.write(f"FAISS index generation completed on gpu {local_rank} (added {self.faiss_index.ntotal} samples to FAISS index)\n")
    sys.stdout.flush()




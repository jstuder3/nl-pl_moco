import faiss
import numpy as np
import torch
import torch.nn.functional as F
from math import floor
import sys
import time


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

    self.faiss_index = faiss.IndexFlatIP(768)

    st = time.time()
    self.faiss_index.add(self.negative_matrix.cpu().numpy())
    print(f"Adding to index took {time.time() - st} seconds")
    print(f"finished on gpu {local_rank}")

    sys.stdout.write(f"FAISS index generation completed on gpu {local_rank} (added {self.faiss_index.ntotal} samples to FAISS index)\n")
    sys.stdout.flush()




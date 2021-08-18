import torch

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

    # [COMPUTE MEAN RECIPROCAL RANK] (MRR)
    # find rank of positive element if the list were sorted (i.e. find number of elements with higher similarity)
    label_list = [logits[i][int(labels[i].item())].item() for i in range(labels.shape[0])]
    label_similarities = torch.tensor(label_list).type_as(docs_list)

    # need to enforce column-wise broadcasting
    ranks = torch.sum(logits >= torch.transpose(label_similarities.view(1, -1), 0, 1), dim=1)  # sum up elements with >= similarity than positive embedding
    mrr = (1 / ranks.shape[0]) * torch.sum(1 / ranks)

    # [COMPUTE TOP5 AND TOP10 ACCURACY]
    # we can reuse the computation for the MRR
    top_5_correct_guesses = torch.sum(ranks <= 5)
    top_5_accuracy = top_5_correct_guesses / docs_list.shape[0]

    top_10_correct_guesses = torch.sum(ranks <= 10)
    top_10_accuracy = top_10_correct_guesses / docs_list.shape[0]

    # [COMPUTE AVERAGE POSITIVE/NEGATIVE COSINE SIMILARITY]
    avg_pos_cos_similarity = torch.mean(label_similarities)

    # sum up all rows, subtract the similarity to the positive sample, then divide by number of samples-1 and finally compute mean over all samples
    avg_neg_cos_similarity = torch.mean((torch.sum(logits, dim=1) - label_similarities) / (code_list.shape[0] - 1))

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

    # like for cosine similarity, compute average of negative similarities
    avg_neg_l2_distance = torch.mean((torch.sum(l2_distance_matrix, dim=1) - label_distances) / (code_list.shape[0] - 1))

    # for good measure
    del label_distances
    del l2_distance_matrix

    # [LOG ALL GENERATED DATA]

    # print to console (mostly for debugging purposes)
    print(f"Validation {substring} MRR (on gpu {self.global_rank}): {mrr:.4f}")
    print(f"Validation {substring} top_1 accuracy (on gpu {self.global_rank}): {top_1_accuracy * 100:.3f}%")
    print(f"Validation {substring} top_5 accuracy (on gpu {self.global_rank}): {top_5_accuracy * 100:.3f}%")
    print(f"Validation {substring} top_10 accuracy (on gpu {self.global_rank}): {top_10_accuracy * 100:.3f}%")
    print(f"Validation {substring} avg_pos_cos_similarity (on gpu {self.global_rank}): {avg_pos_cos_similarity:.6f}")
    print(f"Validation {substring} avg_neg_cos_similarity (on gpu {self.global_rank}): {avg_neg_cos_similarity:.6f}")
    print(f"Validation {substring} avg_pos_l2_distance (on gpu {self.global_rank}): {avg_pos_l2_distance:.6f}")
    print(f"Validation {substring} avg_neg_l2_distance (on gpu {self.global_rank}): {avg_neg_l2_distance:.6f}")

    # log to logger (Tensorboard)
    self.log_dict({f"{base_path_acc}/MRR": mrr, "step": self.current_epoch}, on_epoch=True, sync_dist=True)
    if base_path_acc == "Accuracy_enc/validation":
        self.log("hp_metric", mrr, on_epoch=True, sync_dist=True)
    self.log_dict({f"{base_path_acc}/top_1": top_1_accuracy * 100, "step": self.current_epoch}, on_epoch=True, sync_dist=True)
    self.log_dict({f"{base_path_acc}/top_5": top_5_accuracy * 100, "step": self.current_epoch}, on_epoch=True, sync_dist=True)
    self.log_dict({f"{base_path_acc}/top_10": top_10_accuracy * 100, "step": self.current_epoch}, on_epoch=True, sync_dist=True)
    self.log_dict({f"{base_path_sim}/cosine/positive": avg_pos_cos_similarity, "step": self.current_epoch}, on_epoch=True, sync_dist=True)
    self.log_dict({f"{base_path_sim}/cosine/negative": avg_neg_cos_similarity, "step": self.current_epoch}, on_epoch=True, sync_dist=True)
    self.log_dict({f"{base_path_sim}/l2/positive": avg_pos_l2_distance, "step": self.current_epoch}, on_epoch=True, sync_dist=True)
    self.log_dict({f"{base_path_sim}/l2/negative": avg_neg_l2_distance, "step": self.current_epoch}, on_epoch=True, sync_dist=True)
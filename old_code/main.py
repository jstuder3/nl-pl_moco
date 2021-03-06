# this code will later be adapted for PyTorchLightning. for simplicity, it will be based on just PyTorch for now

import torch
# import transformers
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import time
import argparse
import sys
import re

from utils.data_loading import generateDataLoader

# [HYPERPARAMETERS] (default values, can get overwritten by named call arguments)
num_epochs = 10
batch_size = 8  # see CodeBERT paper
learning_rate = 1e-5  # see CodeBERT paper
temperature = 0.07  # see MoCoV1
queue_size = 32  # limits the number of negative sample batches in the queue
momentum_update_weight = 0.999  # see MoCoV1
model_name = "microsoft/codebert-base"

# limit how much of the total data we use
train_split_size = 10
validation_split_size = 20

validation_batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

output_delay_time = 50

DEBUG_data_skip_interval = 1 # used to skip data during training to get to validation loop faster

# used for tensorboard logging
writer = SummaryWriter()

# [MODEL DEFINITION]
class MoCoModel(nn.Module):
    def __init__(self, max_queue_size, update_weight, model_name):
        super(MoCoModel, self).__init__()
        # initialize all the parts of MoCoV2
        self.encoder = AutoModel.from_pretrained(model_name)
        self.momentum_encoder = AutoModel.from_pretrained(model_name)
        self.queue = []
        self.current_index = 0
        self.max_queue_size = max_queue_size
        self.update_weight = update_weight
        self.encoder_mlp = nn.Sequential( # should there be a relu here or not?
                                         nn.Linear(768, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,
                                                   128))  # 768 is output size of CodeBERT (i.e. BERT_base), 2048 is the hidden layer size MoCoV2 uses and 128 is the output size that SimCLR uses
        self.momentum_encoder_mlp = nn.Sequential( # should there be a relu here or not?
                                                  nn.Linear(768, 2048),
                                                  nn.ReLU(),
                                                  nn.Linear(2048, 128))

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
        encoder_mlp_output = self.encoder_mlp(encoder_mlp_input)
        positive_mlp_output = self.momentum_encoder_mlp(positive_momentum_encoder_mlp_input)
        # only compute the mlp forwards of the queue entries if we're in training
        if not isInference:
            momentum_encoder_mlp_output = torch.tensor([]).to(device)
            # the queue only contains negative samples
            for index, queue_entry in enumerate(self.queue):
                mlp_output = self.momentum_encoder_mlp(queue_entry)
                momentum_encoder_mlp_output = torch.cat((momentum_encoder_mlp_output, mlp_output), axis=0)
            return encoder_mlp_output, positive_mlp_output, momentum_encoder_mlp_output
        else:  # isInference=True
            return encoder_mlp_output, positive_mlp_output

    def update_momentum_encoder(self):
        # update momentum_encoder weights by taking the weighted average of the current weights and the new encoder weights
        # note: need to make sure that this actually works (update: seems to work)
        encoder_params = self.encoder.state_dict()
        for name, param in self.momentum_encoder.named_parameters():
            param = self.update_weight * param + (1 - self.update_weight) * encoder_params[name]

    def getNewestEmbeddings(self):
        # returns the embeddings which have been appended to the queue the most recently
        if len(self.queue) < self.max_queue_size:
            return self.queue[-1]
        else:
            return self.queue[(self.current_index - 1) % self.max_queue_size]  # index is moved forward AFTER updating new entry, so we need to subtract one

    def getIndexOfNewestQueueEntry(self):
        # returns the queue index of the embeddings which have been appended the most recently
        return (self.current_index - 1) % self.max_queue_size

    def replaceOldestQueueEntry(self, newEntry):
        # this function will replace the oldest ("most stale") entry of the queue

        queueIsFull = (len(self.queue) >= self.max_queue_size)
        # if the queue is full, replace the oldest entry
        if queueIsFull:
            self.queue[self.current_index] = newEntry.detach()  # we detach to make sure that we don't waste memory
        # else: queue is not yet full
        else:
            self.queue.append(newEntry.detach())
        self.current_index = (self.current_index + 1) % self.max_queue_size  # potential for off-by-one error

        return queueIsFull  # returning this isn't necessary but might be useful

    def currentQueueSize(self):
        return len(self.queue)


def execute():

    # [GENERATE TRAIN AND VALIDATION LOADER]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader = generateDataLoader("code_search_net", "python", f"train[:{train_split_size}%]", tokenizer, batch_size=batch_size, shuffle=True, augment=True)
    # we don't want to augment the validation set
    val_loader = generateDataLoader("code_search_net", "python", f"validation[:{validation_split_size}%]", tokenizer, batch_size=validation_batch_size, shuffle=False, augment=False)

    # [GENERATE MODEL]
    model = MoCoModel(queue_size, momentum_update_weight, model_name).to(device)

    # [GENERATE OPTIMIZATION STUFF]
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # CodeBERT was pretrained using Adam

    # [TRAINING LOOP]
    training_start_time = time.time()
    consoleOutputTime = time.time()
    queueHasNeverBeenFull = True
    for epoch in range(num_epochs):
        print(f"Starting training of epoch {epoch}...")
        if epoch!=0:
            # generate a newly augmented dataset
            train_loader=generateDataLoader("code_search_net", "python", f"train[:{train_split_size}%]", tokenizer, batch_size=batch_size, shuffle=True, augment=True)
            print(f"Successfully augmented dataset during epoch {epoch}")
        model.train()
        epoch_time = time.time()
        for i, batch in enumerate(train_loader):
            if i % DEBUG_data_skip_interval == 0:  # ONLY USED FOR DEBUGGING PURPOSES TO SKIP TRAINING DATA!!!
                # [PUSH SAMPLES TO GPU]
                doc_samples = {"input_ids": batch["doc_input_ids"].to(device),
                               "attention_mask": batch["doc_attention_mask"].to(device)}
                code_samples = {"input_ids": batch["code_input_ids"].to(device),
                                "attention_mask": batch["code_attention_mask"].to(device)}

                current_batch_size = doc_samples["input_ids"].shape[0]

                # [FORWARD PASS]

                # compute outputs of finetuned CodeBERT encoder and momentum encoder
                encoder_embeddings, positive_momentum_encoder_embeddings = model(doc_samples, code_samples)

                # encoder_mlp contains the mlp output of the queries
                # pos_mlp_emb contains the mlp output of the positive keys
                # neg_mlp_emb contains the mlp output of all of the negative keys in the queue
                encoder_mlp, pos_mlp_emb, neg_mlp_emb = model.mlp_forward(encoder_embeddings,
                                                                          positive_momentum_encoder_embeddings)

                # normalize the length of the embeddings (we want them to be unit vectors for cosine similarity to work correctly)
                encoder_mlp = F.normalize(encoder_mlp, p=2, dim=1)
                if neg_mlp_emb.shape[0] != 0:  # only normalize if non-empty, otherwise normalize() will throw an error
                    neg_mlp_emb = F.normalize(neg_mlp_emb, p=2, dim=1)
                pos_mlp_emb = F.normalize(pos_mlp_emb, p=2, dim=1)

                # [COMPUTE LOSS]

                # clear previous gradients
                optimizer.zero_grad()

                # compute similarity of positive NL/PL pairs
                l_pos = torch.bmm(encoder_mlp.view((current_batch_size, 1, 128)),
                                  pos_mlp_emb.view((current_batch_size, 128, 1)))

                logits = None
                if neg_mlp_emb.shape[0] != 0:
                    # compute similarity of negaitve NL/PL pairs and concatenate with l_pos to get logits
                    l_neg = torch.matmul(encoder_mlp.view((current_batch_size, 128)), torch.transpose(neg_mlp_emb, 0, 1))
                    logits = torch.cat((l_pos.view((current_batch_size, 1)), l_neg), dim=1)
                else:
                    logits = l_pos.view((current_batch_size, 1))

                # labels: l_pos should always contain the smallest values
                labels = torch.tensor([0 for h in range(current_batch_size)]).to(device)  # ugly but does the job

                loss = cross_entropy_loss(logits / temperature, labels)

                if time.time() - consoleOutputTime > output_delay_time:  # output to console if a certain amount of time has passed
                    print(
                        f"Epoch {epoch}, batch {i}/{len(train_loader)}: Loss={loss.item():.4f}, Epoch: {time.time() - epoch_time:.1f}s<<{(1 - ((i + 1) / len(train_loader))) * (time.time() - epoch_time) / ((i + 1) / len(train_loader)):.1f}s, Total: {time.time() - training_start_time:.1f}s")
                    consoleOutputTime = time.time()

                # [BACKPROPAGATION / WEIGHT UPDATES]
                loss.backward()
                optimizer.step()

                # apply a weighted average update on the momentum encoder ("momentum update")
                model.update_momentum_encoder()

                # update the queue
                model.replaceOldestQueueEntry(positive_momentum_encoder_embeddings)
                if queueHasNeverBeenFull and model.currentQueueSize() >= model.max_queue_size:
                    cqs = model.currentQueueSize()
                    print(
                        f"Queue is now full for the first time with {cqs} batches or roughly {batch_size * cqs} samples")
                    queueHasNeverBeenFull = False

                # update tensorboard
                writer.add_scalar("Loss/training", loss.item(), len(train_loader) * epoch + i)

        # [VALIDATION LOOP] (after each epoch)
        model.eval()
        docs_emb_list = torch.tensor([]).to(device)
        code_emb_list = torch.tensor([]).to(device)
        val_start_time = time.time()
        consoleOutputTime = time.time()
        print(f"Starting evaluation after epoch {epoch}, after having seen {epoch * len(train_loader) * batch_size + (i + 1) * batch_size} samples (or {epoch * len(train_loader) + (i + 1)} iterations at batch size {batch_size})...")
        for i, batch in enumerate(val_loader):
            if time.time() - consoleOutputTime > output_delay_time:
                print(
                    f"Validation: {i}/{len(val_loader)}, {time.time() - val_start_time:.1f}s<<{(1 - ((i + 1) / len(val_loader))) * (time.time() - val_start_time) / ((i + 1) / len(val_loader)):.1f}s, Total: {time.time() - training_start_time:.1f}s")
                consoleOutputTime = time.time()

            # [PUSH SAMPLES TO GPU]
            doc_samples = {"input_ids": batch["doc_input_ids"].to(device),
                           "attention_mask": batch["doc_attention_mask"].to(device)}
            code_samples = {"input_ids": batch["code_input_ids"].to(device),
                            "attention_mask": batch["code_attention_mask"].to(device)}

            # [FORWARD PASS]
            with torch.no_grad():
                docs_embeddings, code_embeddings = model(doc_samples, code_samples, isInference=True)  # set to false for experimentation purposes
                # docs_mlp_embeddings, code_mlp_embeddings=model.mlp_forward(docs_embeddings, code_embeddings, isInference=True)
                # docs_mlp_embeddings=F.normalize(docs_mlp_embeddings, p=2, dim=1)
                # code_mlp_embeddings=F.normalize(code_mlp_embeddings, p=2, dim=1)

                # normalize to ensure correct cosine similarity
                docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
                code_embeddings = F.normalize(code_embeddings, p=2, dim=1)


            # we need to find the best match for each NL sample in the entire validation set, so store everything for now
            docs_emb_list = torch.cat((docs_emb_list, docs_embeddings), dim=0)
            code_emb_list = torch.cat((code_emb_list, code_embeddings), dim=0)

        # [COMPARE EVERY QUERY WITH EVERY KEY] (expensive, but necessary for full-corpus accuracy estimation; usually you'd only have one query)

        assert (docs_emb_list.shape == code_emb_list.shape)
        assert (docs_emb_list.shape[1] == 768)  # make sure we use the correct embeddings

        # [COMPUTE PAIRWISE COSINE SIMILARITY MATRIX]
        logits = torch.matmul(docs_emb_list, torch.transpose(code_emb_list, 0, 1)) # warning: size grows quadratically in the number of validation samples (4 GB at 20k samples)

        selection = torch.argmax(logits, dim=1)

        # [COMPUTE TOP1 ACCURACY]
        # the correct guess is always on the diagonal of the logits matrix
        diagonal_label_tensor = torch.tensor([x for x in range(docs_emb_list.shape[0])]).to(device)

        top_1_correct_guesses = torch.sum(selection == diagonal_label_tensor)

        top_1_accuracy = top_1_correct_guesses / docs_emb_list.shape[0]  # accuracy is the fraction of correct guesses

        print(f"Validation top_1 accuracy: {top_1_accuracy * 100:.3f}%")
        writer.add_scalar("Accuracy/validation/top_1", top_1_accuracy * 100, epoch)

        # [COMPUTE MEAN RECIPROCAL RANK] (MRR)
        # find rank of positive element if the list were sorted (i.e. find number of elements with higher similarity)
        diagonal_values = torch.diagonal(logits)
        # need to enforce column-wise broadcasting
        ranks = torch.sum(logits >= torch.transpose(diagonal_values.view(1, -1), 0, 1),
                          dim=1)  # sum up elements with >= similarity than positive embedding
        mrr = (1 / ranks.shape[0]) * torch.sum(1 / ranks)

        print(f"Validation MRR: {mrr:.4f}")
        writer.add_scalar("Accuracy/validation/MRR", mrr, epoch)

        # [COMPUTE TOP5 AND TOP10 ACCURACY]
        # we can reuse the computation for the MRR
        top_5_correct_guesses = torch.sum(ranks <= 5)
        top_10_correct_guesses = torch.sum(ranks <= 10)

        top_5_accuracy = top_5_correct_guesses / docs_emb_list.shape[0]
        top_10_accuracy = top_10_correct_guesses / docs_emb_list.shape[0]
        print(f"Validation top_5 accuracy: {top_5_accuracy * 100:.3f}%")
        print(f"Validation top_10 accuracy: {top_10_accuracy * 100:.3f}%")
        writer.add_scalar("Accuracy/validation/top_5", top_5_accuracy * 100, epoch)
        writer.add_scalar("Accuracy/validation/top_10", top_10_accuracy * 100, epoch)

        # [COMPUTE AVERAGE POSITIVE/NEGATIVE COSINE SIMILARITY]
        avg_pos_cos_similarity = torch.mean(diagonal_values)
        print(f"Validation avg_pos_cos_similarity: {avg_pos_cos_similarity:.6f}")
        writer.add_scalar("Similarity/cosine/positive", avg_pos_cos_similarity, epoch)

        # sum up all rows, subtract the similarity to the positive sample, then divide by number of samples-1 and finally compute mean over all samples
        avg_neg_cos_similarity=torch.mean((torch.sum(logits, dim=1)-diagonal_values)/(docs_emb_list.shape[0]-1))
        print(f"Validation avg_neg_cos_similarity: {avg_neg_cos_similarity:.6f}")
        writer.add_scalar("Similarity/cosine/negative", avg_neg_cos_similarity, epoch)

        # free (potentially) a lot of memory
        del diagonal_values
        del logits

        # [COMPUTE AVERAGE POSITIVE/NEGATIVE L2 DISTANCE]
        l2_distance_matrix=torch.cdist(docs_emb_list, code_emb_list, p=2) # input: [val_set_size, 768], [val_set_size, 768]; output: [val_set_size, val_set_size] pairwise l2 distance # (similarly to logits above, this becomes huge very fast)
        diagonal_l2_distances=torch.diagonal(l2_distance_matrix)

        avg_pos_l2_distance=torch.mean(diagonal_l2_distances)
        print(f"Validation avg_pos_l2_distance: {avg_pos_l2_distance:.6f}")
        writer.add_scalar("Similarity/l2/positive", avg_pos_l2_distance, epoch)

        # like for cosine similarity, compute average of negative similarities
        avg_neg_l2_distance=torch.mean((torch.sum(l2_distance_matrix, dim=1)-diagonal_l2_distances)/(docs_emb_list.shape[0]-1))
        print(f"Validation avg_neg_l2_distance: {avg_neg_l2_distance:.6f}")
        writer.add_scalar("Similarity/l2/negative", avg_neg_l2_distance, epoch)

        # for good measure
        del diagonal_l2_distances
        del l2_distance_matrix

if __name__ == "__main__":
    # [PARSE ARGUMENTS] (if they are given, otherwise keep default value)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--queue_size", type=int)
    parser.add_argument("--momentum_update_weight", type=float)
    parser.add_argument("--train_split_size", type=int)
    parser.add_argument("--validation_split_size", type=int)
    parser.add_argument("--data_skip_interval", type=int)
    parser.add_argument("--output_interval", type=int)
    args = parser.parse_args()

    print(f"[HYPERPARAMETERS] Received as input parameters: {vars(args)}")

    if args.num_epochs != None:
        num_epochs = args.num_epochs
    if args.batch_size != None:
        batch_size = args.batch_size
    if args.learning_rate != None:
        learning_rate = args.learning_rate
    if args.temperature != None:
        temperature = args.temperature
    if args.queue_size != None:
        queue_size = args.queue_size
    if args.momentum_update_weight != None:
        momentum_update_weight = args.momentum_update_weight
    if args.train_split_size != None:
        train_split_size = args.train_split_size
    if args.validation_split_size != None:
        validation_split_size = args.validation_split_size
    if args.data_skip_interval != None:
        DEBUG_data_skip_interval=args.data_skip_interval
    if args.output_interval != None:
        output_delay_time=args.output_interval

    print(f"[HYPERPARAMETERS] Hyperparameters: num_epochs={num_epochs}; batch_size={batch_size}; learning_rate={learning_rate}; temperature={temperature}; queue_size={queue_size}; momentum_update_weight={momentum_update_weight}; train_split_size={train_split_size}; validation_split_size={validation_split_size}; DEBUG_data_skip_interval={DEBUG_data_skip_interval};")

    execute()

    writer.close()

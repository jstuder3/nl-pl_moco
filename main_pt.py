import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
import argparse
from random import randrange

from utils.data_loading import generateDataLoader

# [HYPERPARAMETERS] (default values, can get overwritten by named call arguments)
num_epochs = 10
batch_size = 2  # see CodeBERT paper
learning_rate = 1e-5  # see CodeBERT paper
temperature = 0.07  # see MoCoV1
queue_size = 32  # limits the number of negative sample batches in the queue
momentum_update_weight = 0.999  # see MoCoV1
model_name = "microsoft/codebert-base"

validation_batch_size=32

num_gpus=torch.cuda.device_count()

# limit how much of the total data we use
train_split_size = 1
validation_split_size = 5

class MoCoModelPTL(pl.LightningModule):
    def __init__(self, max_queue_size, update_weight, model_name):
        super().__init__()
        self.automatic_optimization = False # we need that because we have a custom optimization step (with momentum encoder updates)
        # initialize all the parts of MoCoV2
        self.encoder = AutoModel.from_pretrained(model_name)
        self.momentum_encoder = AutoModel.from_pretrained(model_name)
        self.queue = []
        self.current_index = 0
        self.max_queue_size = max_queue_size
        self.update_weight = update_weight
        self.encoder_mlp = nn.Sequential(  # should there be a relu here or not?
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048,
                      128))  # 768 is output size of CodeBERT (i.e. BERT_base), 2048 is the hidden layer size MoCoV2 uses and 128 is the output size that SimCLR uses
        self.momentum_encoder_mlp = nn.Sequential(  # should there be a relu here or not?
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128))

    def forward(self, encoder_input, momentum_encoder_input, isInference=False):

        if not isInference:
            encoder_output = self.encoder(input_ids=encoder_input["input_ids"], attention_mask=encoder_input["attention_mask"])["pooler_output"]
        else:
            with torch.no_grad():
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
        if not isInference:
            encoder_mlp_output = self.encoder_mlp(encoder_mlp_input)
            positive_mlp_output = self.momentum_encoder_mlp(positive_momentum_encoder_mlp_input)
        else:
            with torch.no_grad():
                encoder_mlp_output = self.encoder_mlp(encoder_mlp_input)
                positive_mlp_output = self.momentum_encoder_mlp(positive_momentum_encoder_mlp_input)

        # only compute the mlp forwards of the queue entries if we're in training
        if not isInference:
            momentum_encoder_mlp_output = torch.tensor([])
            # the queue only contains negative samples
            for index, queue_entry in enumerate(self.queue):
                mlp_output = self.momentum_encoder_mlp(queue_entry)
                momentum_encoder_mlp_output = torch.cat((momentum_encoder_mlp_output, mlp_output), axis=0)
            return encoder_mlp_output, positive_mlp_output, momentum_encoder_mlp_output
        else:  # isInference=True
            return encoder_mlp_output, positive_mlp_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)  # CodeBERT was pretrained using Adam
        return optimizer

    def update_momentum_encoder(self):
        # update momentum_encoder weights by taking the weighted average of the current weights and the new encoder weights
        # note: need to make sure that this actually works (update: seems to work)
        # commented out: my versino (probably works, but I don't want to risk anything)
        #encoder_params = self.encoder.state_dict()
        #for name, param in self.momentum_encoder.named_parameters():
        #    param = self.update_weight * param + (1 - self.update_weight) * encoder_params[name]

        # ADAPTED FROM https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/moco/moco2_module.py ! MAY NEED LICENSE TO USE THIS!
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            em = self.update_weight
            param_k.data = param_k.data * em + param_q.data * (1. - em)

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

    def training_step(self, train_batch, batch_idx):
        doc_samples = {"input_ids": train_batch["doc_input_ids"], "attention_mask": train_batch["doc_attention_mask"]}
        code_samples = {"input_ids": train_batch["code_input_ids"], "attention_mask": train_batch["code_attention_mask"]}
        current_batch_size = doc_samples["input_ids"].shape[0]

        # [FORWARD PASS]

        # compute outputs of pretrained CodeBERT encoder and momentum encoder
        encoder_embeddings, positive_momentum_encoder_embeddings = self.forward(doc_samples, code_samples, isInference=False)

        # encoder_mlp contains the mlp output of the queries
        # pos_mlp_emb contains the mlp output of the positive keys
        # neg_mlp_emb contains the mlp output of all of the negative keys in the queue
        encoder_mlp, pos_mlp_emb, neg_mlp_emb = self.mlp_forward(encoder_embeddings, positive_momentum_encoder_embeddings)

        # normalize the length of the embeddings (we want them to be unit vectors for cosine similarity to work correctly)
        encoder_mlp = F.normalize(encoder_mlp, p=2, dim=1)
        if neg_mlp_emb.shape[0] != 0:  # only normalize if non-empty, otherwise normalize() will throw an error
            neg_mlp_emb = F.normalize(neg_mlp_emb, p=2, dim=1)
        pos_mlp_emb = F.normalize(pos_mlp_emb, p=2, dim=1)

        # [COMPUTE LOSS]

        # compute similarity of positive NL/PL pairs
        l_pos = torch.bmm(encoder_mlp.view((current_batch_size, 1, 128)),
                          pos_mlp_emb.view((current_batch_size, 128, 1)))

        logits = None
        if neg_mlp_emb.shape[0] != 0:
            # compute similarity of negaitve NL/PL pairs and concatenate with l_pos to get logits
            l_neg = torch.matmul(encoder_mlp.view((current_batch_size, 128)), torch.transpose(neg_mlp_emb, 0, 1)) #need to check whether if the dimensions here match up because the queue contains negative batches of size num_gpus*batch_size
            logits = torch.cat((l_pos.view((current_batch_size, 1)), l_neg), dim=1)
        else:
            logits = l_pos.view((current_batch_size, 1))

        # labels: l_pos should always contain the smallest values
        labels = torch.tensor([0 for h in range(current_batch_size)]).cuda() # ugly but does the job, also for some reason PTL doesn't automatically move this to the gpu (keeps it on cpu), so we still need .cuda()

        loss = F.cross_entropy(logits / temperature, labels)

        # [BACKPROPAGATION / WEIGHT UPDATES]
        # note: this is PyTorchLightning-specific; we need manual optimization because we have momentum updates
        optimizer=self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        # apply a weighted average update on the momentum encoder ("momentum update")
        self.update_momentum_encoder()

        # update tensorboard
        self.log("Loss/training", loss.item())

        return positive_momentum_encoder_embeddings

    def training_step_end(self, outputs):
        # TODO: check that this works as intended
        print(f"Before: {outputs.shape}")
        outputs = self.all_gather(outputs) # does this do what I expect it to do?
        print(f"After: {outputs.shape}")
        import IPython
        IPython.embed()
        # for out in outputs:
        #     self.replaceOldestQueueEntry(out) # alternatively, we could concatenate all outputs into one tensor and append that tensor to the queue
        concatenated_ouptuts = torch.tensor([]).cuda() # wow... still need to push it to the gpu for this to work smh...
        for out in outputs:
            concatenated_ouptuts = torch.cat((concatenated_ouptuts, out), dim=0) #is this correct? in particular, is dim=0 right?
        self.replaceOldestQueueEntry(concatenated_ouptuts)

    def validation_step(self, val_batch, batch_idx):
        doc_samples = {"input_ids": val_batch["doc_input_ids"], "attention_mask": val_batch["doc_attention_mask"]}
        code_samples = {"input_ids": val_batch["code_input_ids"], "attention_mask": val_batch["code_attention_mask"]}

        # [FORWARD PASS]
        with torch.no_grad():
            docs_embeddings, code_embeddings = self.forward(doc_samples, code_samples, isInference=True)

            # normalize to ensure correct cosine similarity
            docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
            code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        return docs_embeddings, code_embeddings

    # compute validation metrics on just one gpu ("the gatherer")
    def validation_epoch_end(self, outputs):
        # TODO: check that this works as intended

        outputs = self.all_gather(outputs)

        docs_emb_list = torch.tensor([]).cuda() # ... ???
        code_emb_list = torch.tensor([]).cuda()
        for iteration_outs in outputs: # uhm... right? ...
            for out in iteration_outs:
                docs_embeddings, code_embeddings = out
                docs_emb_list = torch.cat((docs_emb_list, docs_embeddings), dim=0)
                code_emb_list = torch.cat((code_emb_list, code_embeddings), dim=0)

        assert (docs_emb_list.shape == code_emb_list.shape)
        assert (docs_emb_list.shape[1] == 768)  # make sure we use the correct embeddings

        # [COMPUTE PAIRWISE COSINE SIMILARITY MATRIX]
        logits = torch.matmul(docs_emb_list, torch.transpose(code_emb_list, 0, 1))  # warning: size grows quadratically in the number of validation samples (4 GB at 20k samples)

        selection = torch.argmax(logits, dim=1)

        # [COMPUTE TOP1 ACCURACY]
        # the correct guess is always on the diagonal of the logits matrix
        diagonal_label_tensor = torch.tensor([x for x in range(docs_emb_list.shape[0])]).cuda()

        top_1_correct_guesses = torch.sum(selection == diagonal_label_tensor)

        top_1_accuracy = top_1_correct_guesses / docs_emb_list.shape[0]  # accuracy is the fraction of correct guesses

        print(f"Validation top_1 accuracy: {top_1_accuracy * 100:.3f}%")
        self.log("Accuracy/validation/top_1", top_1_accuracy * 100)

        # [COMPUTE MEAN RECIPROCAL RANK] (MRR)
        # find rank of positive element if the list were sorted (i.e. find number of elements with higher similarity)
        diagonal_values = torch.diagonal(logits)
        # need to enforce column-wise broadcasting
        ranks = torch.sum(logits >= torch.transpose(diagonal_values.view(1, -1), 0, 1), dim=1)  # sum up elements with >= similarity than positive embedding
        mrr = (1 / ranks.shape[0]) * torch.sum(1 / ranks)

        print(f"Validation MRR: {mrr:.4f}")
        self.log("Accuracy/validation/MRR", mrr)

        # [COMPUTE TOP5 AND TOP10 ACCURACY]
        # we can reuse the computation for the MRR
        top_5_correct_guesses = torch.sum(ranks <= 5)
        top_10_correct_guesses = torch.sum(ranks <= 10)

        top_5_accuracy = top_5_correct_guesses / docs_emb_list.shape[0]
        top_10_accuracy = top_10_correct_guesses / docs_emb_list.shape[0]
        print(f"Validation top_5 accuracy: {top_5_accuracy * 100:.3f}%")
        print(f"Validation top_10 accuracy: {top_10_accuracy * 100:.3f}%")
        self.log("Accuracy/validation/top_5", top_5_accuracy * 100)
        self.log("Accuracy/validation/top_10", top_10_accuracy * 100)

        # [COMPUTE AVERAGE POSITIVE/NEGATIVE COSINE SIMILARITY]
        avg_pos_cos_similarity = torch.mean(diagonal_values)
        print(f"Validation avg_pos_cos_similarity: {avg_pos_cos_similarity:.6f}")
        self.log("Similarity/cosine/positive", avg_pos_cos_similarity)

        # sum up all rows, subtract the similarity to the positive sample, then divide by number of samples-1 and finally compute mean over all samples
        avg_neg_cos_similarity = torch.mean((torch.sum(logits, dim=1) - diagonal_values) / (docs_emb_list.shape[0] - 1))
        print(f"Validation avg_neg_cos_similarity: {avg_neg_cos_similarity:.6f}")
        self.log("Similarity/cosine/negative", avg_neg_cos_similarity)

        # free (potentially) a lot of memory
        del diagonal_values
        del logits

        # [COMPUTE AVERAGE POSITIVE/NEGATIVE L2 DISTANCE]
        l2_distance_matrix = torch.cdist(docs_emb_list, code_emb_list, p=2)  # input: [val_set_size, 768], [val_set_size, 768]; output: [val_set_size, val_set_size] pairwise l2 distance # (similarly to logits above, this becomes huge very fast)
        diagonal_l2_distances = torch.diagonal(l2_distance_matrix)

        avg_pos_l2_distance = torch.mean(diagonal_l2_distances)
        print(f"Validation avg_pos_l2_distance: {avg_pos_l2_distance:.6f}")
        self.log("Similarity/l2/positive", avg_pos_l2_distance)

        # like for cosine similarity, compute average of negative similarities
        avg_neg_l2_distance = torch.mean(
            (torch.sum(l2_distance_matrix, dim=1) - diagonal_l2_distances) / (docs_emb_list.shape[0] - 1))
        print(f"Validation avg_neg_l2_distance: {avg_neg_l2_distance:.6f}")
        self.log("Similarity/l2/negative", avg_neg_l2_distance)

        # for good measure
        del diagonal_l2_distances
        del l2_distance_matrix

def execute():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    val_loader = generateDataLoader("code_search_net", "python", f"validation[:{validation_split_size}%]", tokenizer, batch_size=validation_batch_size, shuffle=False, augment=False)
    torch.cuda.manual_seed_all(randrange(100000))  # synch seed for weight initialization over all gpus (to ensure same initialization for MLP)
    model = MoCoModelPTL(max_queue_size=queue_size, update_weight=momentum_update_weight, model_name=model_name)

    logger = pl.loggers.TensorBoardLogger("runs", name=f"batch_size_{batch_size}-queue_size_{queue_size}-max_epochs_{num_epochs}-train_split_{train_split_size}-val_split_{validation_split_size}-num_gpus_{num_gpus}")

    # IMPORTANT: FOR TESTING ON WINDOWS, USE EITHER DP OR DDP_CPU BECAUSE DDP IS NOT SUPPORTED
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=1, logger=logger, log_gpu_memory="all", fast_dev_run=True, precision=16, accelerator="dp")#, plugins="deepspeed")  # maxepochs=1 because we want to augment after every epoch
    #remove log_gpu_memory and fast_dev_run later because it may slow down training

    for _ in range(num_epochs):
        # generate new augmented dataset
        train_loader = generateDataLoader("code_search_net", "python", f"train[:{train_split_size}%]", tokenizer, batch_size=batch_size, shuffle=True, augment=True)
        # train for one epoch
        trainer.fit(model, train_loader, val_loader)

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
    if args.output_interval != None:
        output_delay_time=args.output_interval

    print(f"[HYPERPARAMETERS] Hyperparameters: num_epochs={num_epochs}; batch_size={batch_size}; learning_rate={learning_rate}; temperature={temperature}; queue_size={queue_size}; momentum_update_weight={momentum_update_weight}; train_split_size={train_split_size}; validation_split_size={validation_split_size};")

    execute()

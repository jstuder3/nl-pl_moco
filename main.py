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

# [HYPERPARAMETERS] (default values, can get overwritten by named call arguments)
num_epochs = 10
batch_size=4 # see CodeBERT paper
learning_rate = 1e-5 # see CodeBERT paper
temperature=0.07 # see MoCoV1
queue_size = 32 # limits the number of negative sample batches in the queue
momentum_update_weight=0.999 # see MoCoV1
model_name = "microsoft/codebert-base"

# limit how much of the total data we use
train_split_size=5
validation_split_size=20

validation_batch_size=32

device="cuda" if torch.cuda.is_available() else "cpu"

output_delay_time=10

# used for tensorboard logging
writer = SummaryWriter()

# [MODEL DEFINITION]
class MoCoModel(nn.Module):
    def __init__(self, max_queue_size, update_weight):
        super(MoCoModel, self).__init__()
        # initialize all the parts of MoCoV2
        self.encoder = AutoModel.from_pretrained(model_name)
        self.momentum_encoder = AutoModel.from_pretrained(model_name)
        self.queue = []
        self.current_index=0
        self.max_queue_size=max_queue_size
        self.update_weight = update_weight
        self.encoder_mlp = nn.Sequential(nn.ReLU(), # should there be a relu here or not?
                                         nn.Linear(768, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, 128)) # 768 is output size of CodeBERT (i.e. BERT_base), 2048 is the hidden layer size MoCoV2 uses and 128 is the output size that SimCLR uses
        self.momentum_encoder_mlp = nn.Sequential(nn.ReLU(), # should there be a relu here or not?
                                                  nn.Linear(768, 2048),
                                                  nn.ReLU(),
                                                  nn.Linear(2048, 128))

    def forward(self, encoder_input, momentum_encoder_input, isInference=False):
        # note entirely sure but I think I may only need the "pooler_output" [bs, 768] and not the "last_hidden_state" [bs, 512, 768]
        encoder_output = self.encoder(input_ids=encoder_input["input_ids"], attention_mask=encoder_input["attention_mask"])["pooler_output"]

        # we save some memory by immediately detaching the momentum encoder output.
        # we don't need the computation graph of that because we won't backprop through the momentum encoder
        if isInference: #use the encoder
            momentum_encoder_output = self.encoder(input_ids=momentum_encoder_input["input_ids"], attention_mask=momentum_encoder_input["attention_mask"])["pooler_output"].detach()
        else: #use the momentum encoder
            momentum_encoder_output = self.momentum_encoder(input_ids=momentum_encoder_input["input_ids"], attention_mask=momentum_encoder_input["attention_mask"])["pooler_output"].detach()

        return encoder_output, momentum_encoder_output

    def mlp_forward(self, encoder_mlp_input, positive_momentum_encoder_mlp_input, isInference=False):
        encoder_mlp_output = self.encoder_mlp(encoder_mlp_input)
        positive_mlp_output = self.momentum_encoder_mlp(positive_momentum_encoder_mlp_input)
        #only compute the mlp forwards of the queue entries if we're in training
        if not isInference:
            momentum_encoder_mlp_output = torch.tensor([]).to(device)
            # the queue only contains negative samples
            for index, queue_entry in enumerate(self.queue):
                mlp_output = self.momentum_encoder_mlp(queue_entry)
                momentum_encoder_mlp_output = torch.cat((momentum_encoder_mlp_output, mlp_output), axis=0)
            # momentum_encoder_mlp_output = []
            # for queue_entry in self.queue:
            #     momentum_encoder_mlp_output.append(self.momentum_encoder_mlp(queue_entry))
            return encoder_mlp_output, positive_mlp_output, momentum_encoder_mlp_output
        else: #isInference=True
            return encoder_mlp_output, positive_mlp_output

    def update_momentum_encoder(self):
        # update momentum_encoder weights by taking the weighted average of the current weights and the new encoder weights
        # note: need to make sure that this actually works (update: seems to work)
        encoder_params = self.encoder.state_dict()
        for name, param in self.momentum_encoder.named_parameters():
            param=self.update_weight * param + (1-self.update_weight)*encoder_params[name]

    def getNewestEmbeddings(self):
        # returns the embeddings which have been appended to the queue the most recently
        if len(self.queue)<self.max_queue_size:
            return self.queue[-1]
        else:
            return self.queue[(self.current_index-1)%self.max_queue_size] # index is moved forward AFTER updating new entry, so we need to subtract one

    def getIndexOfNewestQueueEntry(self):
        # returns the queue index of the embeddings which have been appended the most recently
        return (self.current_index-1)%self.max_queue_size

    def replaceOldestQueueEntry(self, newEntry):
        # this function will replace the oldest ("most stale") entry of the queue

        queueIsFull = (len(self.queue) >= self.max_queue_size)
        # if the queue is full, replace the oldest entry
        if queueIsFull:
            self.queue[self.current_index] = newEntry.detach()  # we detach to make sure that we don't waste memory
        # else: queue is not yet full
        else:
            self.queue.append(newEntry.detach())
        self.current_index = (self.current_index + 1) % self.max_queue_size # potential for off-by-one error

        return queueIsFull # returning this isn't necessary but might be useful

    def currentQueueSize(self):
        return len(self.queue)

# [DATASET DEFINITION]
class CodeSearchNetDataset(torch.utils.data.Dataset):
    def __init__(self, doc_tokens, code_tokens):
        self.doc_tokens = doc_tokens
        self.code_tokens = code_tokens

    def __getitem__(self, index):
        # item=[]
        # item.append({key: torch.tensor(val[index]) for key, val in self.doc_tokens.items()})
        # item.append({key: torch.tensor(val[index]) for key, val in self.code_tokens.items()})
        #item1 = {"doc_"+key: torch.tensor(val[index]) for key, val in self.doc_tokens.items()}
        #item2 = {"code_"+key: torch.tensor(val[index]) for key, val in self.code_tokens.items()}
        #item = {**item1, **item2} # only requires python 3.5+
        item = {"doc_" + key: torch.tensor(val[index]) for key, val in self.doc_tokens.items()}
        item = item | {"code_"+key: torch.tensor(val[index]) for key, val in self.code_tokens.items()} #this syntax requires python 3.9+
        # item = {"input_ids": self.doc_tokens["input_ids"][index], "attention_mask": self.doc_tokens["attention_mask"][index], "label": self.code_tokens["input_ids"][index], "label_attention_mask": self.code_tokens["attention_mask"][index]}
        return item

    def __len__(self):
        return len(self.doc_tokens["input_ids"])

# takes as input a dict that has a key "func_documentation_string", shortens that to the first paragraph and puts the result under the key "func_documentation_string_shortened"
def shorten_data(dict):
    shortened_doc = " ".join(dict["func_documentation_string"].partition("\n\n")[0].split()) #shortens to the first paragraph and removes all "\n" and multi-whitespaces from the remainders
    dict["func_documentation_string_shortened"] = shortened_doc
    return dict

# takes as input a dict that has a key "func_code_tokens", concatenates all the tokens and puts them into the key "func_code_tokens_concatenated"
# a bit simpler than filtering "func_code_string" but might cause issues later because the output really isn't "perfect"
# in the sense that it contains many unnecessary whitespaces
def joinCodeTokens(dict):
    concatenated_tokens = " ".join(dict["func_code_tokens"])
    dict["func_code_string_cleaned"]=concatenated_tokens
    return dict

def execute():

    # [LOAD DATA]
    train_data_raw = load_dataset("code_search_net", "python", split=f"train[:{train_split_size}%]") #change "python" to "all" or any of the other languages to get different subset
    # this returns a dictionary (for split "train", "validation", "test" or all of them if none selected) with several keys, but we only really care about "func_code_tokens" and
    # "func_documentation_tokens", which both return a list of tokens (strings)

    val_data_raw = load_dataset("code_search_net", "python", split=f"validation[:{validation_split_size}%]")

    # [FILTERING AND PREPROCESSING]

    # the dataset is already prefiltered. the only thing we need to do is to shorten documentations to the first paragraph.
    # NOTE: we can't just use the pre-tokenized column in the dataset because it does not contain empty spaces, thus potentially losing information

    train_data_raw=train_data_raw.map(shorten_data)
    train_data_raw=train_data_raw.map(joinCodeTokens)

    val_data_raw=val_data_raw.map(shorten_data)
    val_data_raw=val_data_raw.map(joinCodeTokens)

    # [TOKENIZE DATA]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_docs_tokens = tokenizer(train_data_raw["func_documentation_string_shortened"], truncation=True, padding="max_length")
    train_code_tokens = tokenizer(train_data_raw["func_code_string_cleaned"], truncation=True, padding="max_length")

    val_docs_tokens = tokenizer(val_data_raw["func_documentation_string_shortened"], truncation=True, padding="max_length")
    val_code_tokens = tokenizer(val_data_raw["func_code_string_cleaned"], truncation=True, padding="max_length")

    # [CREATE DATASET OBJECTS]

    train_dataset = CodeSearchNetDataset(train_docs_tokens, train_code_tokens)

    val_dataset = CodeSearchNetDataset(val_docs_tokens, val_code_tokens)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # this expects an entry in the dict that is called "label"
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=False)

    # [GENERATE MODEL]
    model = MoCoModel(queue_size, momentum_update_weight).to(device)

    # [GENERATE OPTIMIZATION STUFF]
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #CodeBERT was pretrained using Adam

    # [TRAINING LOOP]
    training_start_time=time.time()
    consoleOutputTime=time.time()
    queueHasNeverBeenFull=True
    for epoch in range(num_epochs):
        print(f"Starting training of epoch {epoch}...")
        model.train()
        epoch_time=time.time()
        for i, batch in enumerate(train_loader):
            if i%1==0: # ONLY USED FOR DEBUGGING PURPOSES TO SKIP TRAINING DATA!!!
                # [PUSH SAMPLES TO GPU]
                doc_samples = {"input_ids": batch["doc_input_ids"].to(device), "attention_mask": batch["doc_attention_mask"].to(device)}
                code_samples = {"input_ids": batch["code_input_ids"].to(device), "attention_mask": batch["code_attention_mask"].to(device)}

                current_batch_size = doc_samples["input_ids"].shape[0]

                # [FORWARD PASS]

                # compute outputs of finetuned CodeBERT encoder and momentum encoder
                encoder_embeddings, positive_momentum_encoder_embeddings = model(doc_samples, code_samples)

                # encoder_mlp contains the mlp output of the queries
                # pos_mlp_emb contains the mlp output of the positive keys
                # neg_mlp_emb contains the mlp output of all of the negative keys in the queue
                encoder_mlp, pos_mlp_emb, neg_mlp_emb = model.mlp_forward(encoder_embeddings, positive_momentum_encoder_embeddings)

                # normalize the length of the embeddings (we want them to be unit vectors for cosine similarity to work correctly)
                encoder_mlp = F.normalize(encoder_mlp, p=2, dim=1)
                if neg_mlp_emb.shape[0] != 0: # only normalize if non-empty, otherwise normalize() will throw an error
                    neg_mlp_emb = F.normalize(neg_mlp_emb, p=2, dim=1)
                pos_mlp_emb = F.normalize(pos_mlp_emb, p=2, dim=1)

                # [COMPUTE LOSS]

                # clear previous gradients
                optimizer.zero_grad()

                # compute similarity of positive NL/PL pairs
                l_pos = torch.bmm(encoder_mlp.view((current_batch_size, 1, 128)), pos_mlp_emb.view((current_batch_size, 128, 1)))

                logits = None
                if neg_mlp_emb.shape[0] != 0:
                    # compute similarity of negaitve NL/PL pairs and concatenate with l_pos to get logits
                    l_neg = torch.matmul(encoder_mlp.view((current_batch_size, 128)), torch.transpose(neg_mlp_emb, 0, 1))
                    logits = torch.cat((l_pos.view((current_batch_size, 1)), l_neg), dim=1)
                else:
                    logits = l_pos.view((current_batch_size, 1))

                # torch.zeros((current_batch_size, 1), dtype=torch.LongTensor).to(device) #this function has only very limited
                # function signature overloads, and just converting to torch.LongTensor throws an error somehow...

                # labels: l_pos should always contain the smallest values
                labels = torch.tensor([0 for h in range(current_batch_size)]).to(device)  # ugly but does the job

                loss = cross_entropy_loss(logits/temperature, labels)

                if time.time()-consoleOutputTime > output_delay_time: # output to console if a certain amount of time has passed
                    print(f"Epoch {epoch}, batch {i}/{len(train_loader)}: Loss={loss.item():.4f}, Epoch: {time.time()-epoch_time:.1f}s<<{(1-((i+1)/len(train_loader)))*(time.time()-epoch_time)/((i+1)/len(train_loader)):.1f}s, Total: {time.time()-training_start_time:.1f}s")
                    consoleOutputTime=time.time()

                # [BACKPROPAGATION / WEIGHT UPDATES]
                loss.backward()
                optimizer.step()

                # apply a weighted average update on the momentum encoder ("momentum update")
                model.update_momentum_encoder()

                # update the queue
                model.replaceOldestQueueEntry(positive_momentum_encoder_embeddings)
                if queueHasNeverBeenFull and model.currentQueueSize()>=model.max_queue_size:
                    cqs=model.currentQueueSize()
                    print(f"Queue is now full for the first time with {cqs} batches or roughly {batch_size*cqs} samples")
                    queueHasNeverBeenFull=False

                # update tensorboard
                writer.add_scalar("Loss/training", loss.item(), len(train_loader)*epoch + i)

        # [VALIDATION LOOP] (after each epoch)
        model.eval()
        docs_emb_list=torch.tensor([]).to(device)
        code_emb_list=torch.tensor([]).to(device)
        val_start_time=time.time()
        consoleOutputTime=time.time()
        print(f"Starting evaluation after epoch {epoch}, after having seen {epoch*len(train_loader)*batch_size+(i+1)*batch_size} samples (or {epoch*len(train_loader)+(i+1)} iterations at batch size {batch_size})...")
        for i, batch in enumerate(val_loader):
            if time.time()-consoleOutputTime>output_delay_time:
                print(f"Validation: {i}/{len(val_loader)}, {time.time()-val_start_time:.1f}s<<{(1-((i+1)/len(val_loader)))*(time.time()-val_start_time)/((i+1)/len(val_loader)):.1f}s, Total: {time.time()-training_start_time:.1f}s")
                consoleOutputTime=time.time()

            # [PUSH SAMPLES TO GPU]
            doc_samples = {"input_ids": batch["doc_input_ids"].to(device),
                           "attention_mask": batch["doc_attention_mask"].to(device)}
            code_samples = {"input_ids": batch["code_input_ids"].to(device),
                            "attention_mask": batch["code_attention_mask"].to(device)}

            # [FORWARD PASS]
            with torch.no_grad():
                docs_embeddings, code_embeddings = model(doc_samples, code_samples, isInference=False) #set to false for experimentation purposes
                docs_mlp_embeddings, code_mlp_embeddings=model.mlp_forward(docs_embeddings, code_embeddings, isInference=True)
                #normalize to ensure correct cosine similarity
                #docs_embeddings=F.normalize(docs_embeddings, p=2, dim=1)
                #code_embeddings=F.normalize(code_embeddings, p=2, dim=1)
                docs_mlp_embeddings=F.normalize(docs_mlp_embeddings, p=2, dim=1)
                code_mlp_embeddings=F.normlaize(code_mlp_embeddings, p=2, dim=1)

            # we need to find the best match for each NL sample in the entire validation set, so store everything for now
            # docs_emb_list.append(docs_embeddings)
            # code_emb_list.append(code_emb_list)
            #docs_emb_list=torch.cat((docs_emb_list, docs_embeddings), dim=0)
            #code_emb_list=torch.cat((code_emb_list, code_embeddings), dim=0)
            docs_emb_list = torch.cat((docs_emb_list, docs_mlp_embeddings), dim=0)
            code_emb_list=torch.cat((code_emb_list, code_mlp_embeddings), dim=0)

        # [COMPARE EVERY QUERY WITH EVERY KEY] (expensive, but necessary for full-corpus accuracy estimation; usually you'd only have one query)

        assert(docs_emb_list.shape==code_emb_list.shape)
        #assert(docs_emb_list.shape[1]==768) # make sure we use the correct embeddings

        logits=torch.matmul(docs_emb_list, torch.transpose(code_emb_list, 0, 1))

        selection = torch.argmax(logits, dim=1)

        # the correct guess is always on the diagonal of the logits matrix
        label = torch.tensor([x for x in range(docs_emb_list.shape[0])]).to(device)

        top_1_correct_guesses = torch.sum(selection==label)

        top_1_accuracy = top_1_correct_guesses/docs_emb_list.shape[0] # accuracy is the fraction of correct guesses

        print(f"Validation accuracy: {top_1_accuracy*100:.3f}%")
        writer.add_scalar("Accuracy/validation", top_1_accuracy*100, epoch)

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
    args=parser.parse_args()

    print(f"[HYPERPARAMETERS] Received as input parameters: {vars(args)}")

    if args.num_epochs!=None:
        num_epochs=args.num_epochs
    if args.batch_size!=None:
        batch_size=args.batch_size
    if args.learning_rate!=None:
        learning_rate=args.learning_rate
    if args.temperature!=None:
        temperature=args.temperature
    if args.queue_size!=None:
        queue_size=args.queue_size
    if args.momentum_update_weight != None:
        momentum_update_weight = args.momentum_update_weight
    if args.train_split_size!=None:
        train_split_size=args.train_split_size
    if args.validation_split_size!=None:
        validation_split_size=args.validation_split_size

    print(f"[HYPERPARAMETERS] Hyperparameters: num_epochs={num_epochs}; batch_size={batch_size}; learning_rate={learning_rate}; temperature={temperature}; queue_size={queue_size}; momentum_update_weight={momentum_update_weight}; train_split_size={train_split_size}; validation_split_size={validation_split_size}")

    execute()


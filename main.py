# this code will later be adapted for PyTorchLightning. for simplicity, it will be based on just PyTorch for now

import torch
# import transformers
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AdamW
from datasets import load_dataset

# [HYPERPARAMETERS]
num_epochs = 1
learning_rate = 5e-4 # see CodeBERT paper
batch_size=2 # see CodeBERT paper
temperature=0.07 # see MoCoV1
queue_size = 200 # limits the number of negative sample batches in the queue
momentum_update_weight=0.999 # see MoCoV1
model_name = "microsoft/codebert-base"

device="cuda" if torch.cuda.is_available() else "cpu"

#[MODEL DEFINITION]
class MoCoModel(nn.Module):
    def __init__(self, max_queue_size, update_weight):
        super(MoCoModel, self).__init__()
        #initialize all the parts of MoCoV2
        self.encoder = AutoModel.from_pretrained(model_name)
        self.momentum_encoder = AutoModel.from_pretrained(model_name)
        self.queue = []
        self.current_index=0
        self.max_queue_size=max_queue_size
        self.update_weight = update_weight
        self.encoder_mlp = nn.Sequential(nn.Linear(768, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, 128)) # 768 is output size of CodeBERT (i.e. BERT_base), 2048 is the hidden layer size MoCoV2 uses and 128 is the output size that SimCLR uses
        self.momentum_encoder_mlp = nn.Sequential(nn.Linear(768, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, 128))

    def forward(self, encoder_input, momentum_encoder_input):
        # note entirely sure but I think I may only need the "pooler_output" [bs, 768] and not the "last_hidden_state" [bs, 512, 768]
        encoder_output = self.encoder(input_ids=encoder_input["input_ids"], attention_mask=encoder_input["attention_mask"])["pooler_output"]

        # we save some memory by immediately detaching the momentum encoder output.
        # we don't need the computation graph of that because we won't backprop through the momentum encoder
        momentum_encoder_output = self.momentum_encoder(input_ids=momentum_encoder_input["input_ids"], attention_mask=momentum_encoder_input["attention_mask"])["pooler_output"].detach()

        return encoder_output, momentum_encoder_output

    def mlp_forward(self, encoder_mlp_input, positive_momentum_encoder_mlp_input):
        encoder_mlp_output = self.encoder_mlp(encoder_mlp_input)
        positive_mlp_output = self.momentum_encoder_mlp(positive_momentum_encoder_mlp_input)

        momentum_encoder_mlp_output = torch.tensor([]).to(device)
        # the queue only contains negative samples
        for index, queue_entry in enumerate(self.queue):
            mlp_output = self.momentum_encoder_mlp(queue_entry)
            momentum_encoder_mlp_output = torch.cat((momentum_encoder_mlp_output, mlp_output), axis=0)
        # momentum_encoder_mlp_output = []
        # for queue_entry in self.queue:
        #     momentum_encoder_mlp_output.append(self.momentum_encoder_mlp(queue_entry))
        return encoder_mlp_output, positive_mlp_output, momentum_encoder_mlp_output

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

# [LOAD DATA]
dataset = load_dataset("code_search_net", "python", split="train[:5%]") #change "python" to "all" or any of the other languages to get different subset
# this returns a dictionary (for split "train", "validation", "test" or all of them if none selected) with several keys, but we only really care about "func_code_tokens" and
# "func_documentation_tokens", which both return a list of tokens (strings)

# [FILTERING AND PREPROCESSING]

# the dataset is already prefiltered. the only thing we need to do is to shorten documentations to the first paragraph.
# NOTE: we can't just use the pre-tokenized column in the dataset because it does not contain empty spaces, thus potentially losing information

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

dataset=dataset.map(shorten_data)
dataset=dataset.map(joinCodeTokens)

# [TOKENIZE DATA]
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_docs_tokens = tokenizer(dataset["func_documentation_string_shortened"], truncation=True, padding=True)
train_code_tokens = tokenizer(dataset["func_code_string_cleaned"], truncation=True, padding=True)

# [CREATE TRAIN DATASET OBJECT]

class CodeSearchNetDataset(torch.utils.data.Dataset):
    def __init__(self, doc_tokens, code_tokens):
        self.doc_tokens = doc_tokens
        self.code_tokens = code_tokens

    def __getitem__(self, index):
        # item=[]
        # item.append({key: torch.tensor(val[index]) for key, val in self.doc_tokens.items()})
        # item.append({key: torch.tensor(val[index]) for key, val in self.code_tokens.items()})
        item = {"doc_"+key: torch.tensor(val[index]) for key, val in self.doc_tokens.items()}
        item = item | {"code_"+key: torch.tensor(val[index]) for key, val in self.code_tokens.items()} #this syntax requires python 3.9+
        # item = {"input_ids": self.doc_tokens["input_ids"][index], "attention_mask": self.doc_tokens["attention_mask"][index], "label": self.code_tokens["input_ids"][index], "label_attention_mask": self.code_tokens["attention_mask"][index]}
        return item

    def __len__(self):
        return len(self.doc_tokens["input_ids"])


train_dataset = CodeSearchNetDataset(train_docs_tokens, train_code_tokens)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#this expects an entry in the dict that is called "label"

#[GENERATE MODEL]
model = MoCoModel(queue_size, momentum_update_weight).to(device)

#[GENERATE OPTIMIZATION STUFF]
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #CodeBERT was pretrained using Adam

# [TRAINING LOOP]
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):

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
        print(loss.item())

        # [BACKPROPAGATION / WEIGHT UPDATES]
        loss.backward()
        optimizer.step()

        # apply a weighted average update on the momentum encoder ("momentum update")
        model.update_momentum_encoder()

        #update the queue
        model.replaceOldestQueueEntry(positive_momentum_encoder_embeddings)

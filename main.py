#this code will later be adapted for PyTorchLightning. for simplicity, it will be based on just PyTsorch for now

import torch
#import transformers
from torch import nn
from transformers import AutoModel, AutoTokenizer, AdamW
from datasets import load_dataset

num_epochs = 1
learning_rate = 5e-4 #see CodeBERT paper
batch_size=2048 #see CodeBERT paper
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
                                         nn.Linear(2048, 128)) #768 is output size of CodeBERT (i.e. BERT_base), 2048 is the hidden layer size MoCoV2 uses and 128 is the output size that SimCLR uses
        self.momentum_encoder_mlp = nn.Sequential(nn.Linear(768, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, 128))

    def forward(self, encoder_input, momentum_encoder_input):
        encoder_output = self.encoder(encoder_input)
        momentum_encoder_output = self.momentum_encoder(momentum_encoder_input)

        #if the queue is full, replace the oldest entry
        if (len(self.queue)>=self.max_queue_size):
            self.queue[self.current_index]=momentum_encoder_output
            self.current_index=(self.current_index+1)%self.max_queue_size #I smell an off-by-one error
        else: #queue is not yet full
            self.queue.append(momentum_encoder_output)

        return encoder_output #we will manually access the queue later, so we only need to return this for now

    def mlp_forward(self, encoder_mlp_input, momentum_encoder_mlp_input):
        encoder_mlp_output = self.encoder_mlp(encoder_mlp_input)
        momentum_encoder_mlp_output = self.momentum_encoder_mlp(momentum_encoder_mlp_input)  # if feeding self.queue even works directly; note: we will need to update the right side up to the MLP since the two MLPs don't share weights
        return encoder_mlp_output, momentum_encoder_mlp_output

    def update_momentum_encoder(self):
        #update momentum_encoder weights by taking the weighted average of the current weights and the new encoder weights
        self.momentum_encoder.copy(self.update_weight*self.momentum_encoder.parameters() + (1-self.update_weight)*self.encoder.parameters())
        #need to verify that this does the intended thing

#[LOAD DATA]
dataset = load_dataset("code_search_net", "python", split="train[:5%]") #change "python" to "all" or any of the other languages to get different subset
#this returns a dictionary (for split "train", "validation", "test" or all of them if none selected) with several keys, but we only really care about "func_code_tokens" and
#"func_documentation_tokens", which both return a list of tokens (strings)

#[FILTERING AND PREPROCESSING]

#the dataset is already prefiltered. the only thing we need to do is to shorten documentations to the first paragraph.
#NOTE: we can't just use the pre-tokenized column in the dataset because it does not contain empty spaces, thus potentially losing information

#takes as input a dict that has a key "func_documentation_string", shortens that to the first paragraph and puts the result under the key "func_documentation_string_shortened"
def shorten_data(dict):
    shortened_doc = " ".join(dict["func_documentation_string"].partition("\n\n")[0].split()) #shortens to the first paragraph and removes all "\n" and multi-whitespaces from the remainders
    dict["func_documentation_string_shortened"] = shortened_doc
    return dict

#takes as input a dict that has a key "func_code_tokens", concatenates all the tokens and puts them into the key "func_code_tokens_concatenated"
#a bit simpler than filtering "func_code_string" but might cause issues later because the output really isn't "perfect"
#in the sense that it contains many unnecessary whitespaces
def joinCodeTokens(dict):
    concatenated_tokens = " ".join(dict["func_code_tokens"])
    dict["func_code_string_cleaned"]=concatenated_tokens
    return dict

#dataset=dataset.add_column("func_documentation_string_shortened", ["" for i in range(len(list(dataset)))]) #apparently not necessary
dataset=dataset.map(shorten_data)
dataset=dataset.map(joinCodeTokens)



#[TOKENIZE DATA]
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_docs_tokens = tokenizer(dataset["func_documentation_string_shortened"], truncation=True, padding=True)
train_code_tokens = tokenizer(dataset["func_code_string_cleaned"], truncation=True, padding=True)

#[CREATE TRAIN DATASET OBJECT]

class CodeSearchNetDataset(torch.utils.data.Dataset):
    def __init__(self, doc_tokens, code_tokens):
        self.doc_tokens = doc_tokens
        self.code_tokens = code_tokens

    def __getitem__(self, index):
        #item=[]
        #item.append({key: torch.tensor(val[index]) for key, val in self.doc_tokens.items()})
        #item.append({key: torch.tensor(val[index]) for key, val in self.code_tokens.items()})
        item = {"doc_"+key: torch.tensor(val[index]) for key, val in self.doc_tokens.items()}
        item = item | {"code_"+key: torch.tensor(val[index]) for key, val in self.code_tokens.items()} #requires python 3.9+
        #item = {"input_ids": self.doc_tokens["input_ids"][index], "attention_mask": self.doc_tokens["attention_mask"][index], "label": self.code_tokens["input_ids"][index], "label_attention_mask": self.code_tokens["attention_mask"][index]}
        return item

    def __len__(self):
        return len(self.doc_tokens["input_ids"])


train_dataset = CodeSearchNetDataset(train_docs_tokens, train_code_tokens)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#this expects an entry in the dict that is called "label"

#[GENERATE MODEL]
model = MoCoModel(8, 0.999).to(device)

#[GENERATE OPTIMIZATION STUFF]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #CodeBERT was pretrained using Adam

#[TRAINING LOOP]
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        doc_samples = {"input_ids": batch["doc_input_ids"], "attention_mask": batch["doc_attention_mask"]}
        code_samples = {"input_ids": batch["code_input_ids"], "attention_mask": batch["code_attention_mask"]}
        x=1

        optimizer.zero_grad()


#[TRAINING LOOP]






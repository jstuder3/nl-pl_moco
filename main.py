#this code will later be adapted for PyTorchLightning. for simplicity, it will be based on just PyTsorch for now

#import torch
#import transformers
from torch import nn
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

model_name = "microsoft/CodeBERT"

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
dataset = load_dataset("code_serach_net", "python", split="train[:10%]") #change "python" to "all" or any of the other languages to get different subset
#this returns a dictionary (for split "train", "validation", "test" or all of them if none selected) with several keys, but we only really care about "func_code_tokens" and
#"func_documentation_tokens", which both return a list of tokens (strings)

#[TOKENIZE DATA]
tokenizer = AutoTokenizer.from_pretrained(model_name)

#[TRAINING LOOP]






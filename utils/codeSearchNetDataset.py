import torch

# [DATASET DEFINITION]
class CodeSearchNetDataset(torch.utils.data.Dataset):
    def __init__(self, doc_tokens, code_tokens):
        self.doc_tokens = doc_tokens
        self.code_tokens = code_tokens

    def __getitem__(self, index):
        if type(self.doc_tokens["input_ids"])==torch.Tensor:
            item = {"doc_" + key: val[index] for key, val in self.doc_tokens.items()}
            item = item | {"code_" + key: val[index] for key, val in self.code_tokens.items()}  # this syntax requires python 3.9+
            item = item | {"index": index}
        else:
            item = {"doc_" + key: torch.tensor(val[index]) for key, val in self.doc_tokens.items()}
            item = item | {"code_" + key: torch.tensor(val[index]) for key, val in self.code_tokens.items()}  # this syntax requires python 3.9+
        return item

    def __len__(self):
        return len(self.doc_tokens["input_ids"])

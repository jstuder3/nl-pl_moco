
import torch
import torch.nn.functional as F
import pytorch_lightning as pt
from utils.multimodal_data_loading import generateDataLoader
from transformers import AutoTokenizer, AutoModel
import argparse
import sys

from xMoCo_pl import xMoCoModelPTL

class xMoCoVanillaModelPTL(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args=args

        self.batch_size = args.effective_batch_size
        self.num_gpus = args.num_gpus
        self.num_hard_negatives = self.args.num_hard_negatives

        self.docs_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_tokenizer = self.docs_tokenizer

        self.docs_fast_encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.code_fast_encoder = self.docs_fast_encoder

@torch.no_grad()
def generateEmbeddings(model, dataloader, subset):
    #forwards the given dataloader through the code_fast_encoder to obtain embeddings
    docs_embeddings_tensor=torch.tensor([]).cuda()
    code_embeddings_tensor=torch.tensor([]).cuda()

    if model.docs_fast_encoder.device==torch.device("cpu"):
        model.docs_fast_encoder.cuda()

    if model.code_fast_encoder.device==torch.device("cpu"): #if it's on cpu, put it on gpu
        model.code_fast_encoder.cuda()

    if model.docs_fast_encoder.training:
        model.docs_fast_encoder.eval()

    if model.code_fast_encoder.training:
        model.code_fast_encoder.eval()

    for i, batch in enumerate(dataloader):
        docs_samples = {"input_ids": batch["doc_input_ids"].cuda(), "attention_mask": batch["doc_attention_mask"].cuda()}
        code_samples = {"input_ids": batch["code_input_ids"].cuda(), "attention_mask": batch["code_attention_mask"].cuda()}
        docs_embeddings = model.docs_fast_encoder(input_ids=docs_samples["input_ids"], attention_mask=docs_samples["attention_mask"])["pooler_output"]
        code_embeddings = model.code_fast_encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
        docs_embeddings = F.normalize(docs_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)
        docs_embeddings_tensor = torch.cat((docs_embeddings_tensor, docs_embeddings), dim=0)
        code_embeddings_tensor = torch.cat((code_embeddings_tensor, code_embeddings), dim=0)

        sys.stdout.write(f"\rForwarding samples ({subset} subset): {i+1} / {len(dataloader)} ({(i+1)/len(dataloader)*100:.2f}%)")
        sys.stdout.flush()

    return docs_embeddings_tensor, code_embeddings_tensor

def execute(args):

    dataloader=None

    try:
        docs_untrained = torch.load(f"cache/untrained_ruby_docs_train.pt")
        code_untrained = torch.load(f"cache/untrained_ruby_code_train.pt")

    except:
        model = xMoCoVanillaModelPTL(args)

        dataloader = generateDataLoader("ruby", "train", model.docs_tokenizer, model.code_tokenizer, args)

        docs_untrained, code_untrained = generateEmbeddings(model, dataloader, "untrained_train")
        docs_untrained=docs_untrained.half()
        code_untrained=code_untrained.half()
        torch.save(docs_untrained, f"cache/untrained_ruby_docs_train.pt")
        torch.save(code_untrained, f"cache/untrained_ruby_code_train.pt")

    try:
        docs_trained = torch.load(f"cache/ruby_docs_train.pt")
        code_trained = torch.load(f"cache/ruby_train.pt")
    except:
        model = xMoCoModelPTL.load_from_checkpoint(f"checkpoints/ruby.ckpt")

        if dataloader==None:
            dataloader = generateDataLoader("ruby", "train", model.docs_tokenizer, model.code_tokenizer, args)

        docs_untrained, code_untrained = generateEmbeddings(model, dataloader, "trained_train")
        docs_untrained = docs_untrained.half()
        code_untrained = code_untrained.half()
        torch.save(docs_untrained, f"cache/ruby_docs_train.pt")
        torch.save(code_untrained, f"cache/ruby_train.pt")

    print(f"Everything preprocessed or loaded successfully!")

    print(f"Now we could do dimensionality reduction, but it's not implemented yet...")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_folder", type=str, default="datasets/CodeSearchNet")
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--debug_data_skip_interval", type=int, default=1)
    parser.add_argument("--always_use_full_val", default=True)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_hard_negatives", type=int, default=0)
    args = parser.parse_args()

    execute(args)




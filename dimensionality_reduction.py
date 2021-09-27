
import torch
import torch.nn.functional as F
import pytorch_lightning as pt
from utils.multimodal_data_loading import generateDataLoader
from transformers import AutoTokenizer, AutoModel
import argparse
import sys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def downprojection(docs, code, n_pca_samples=1000, n_tsne_samples=200, pca_dimension=50, tsne_iterations=500):
    rndperm = np.random.permutation(docs.shape[0])

    pca = PCA(n_components=pca_dimension)
    data_joined = np.concatenate((docs, code), axis=0)
    pca_projection = pca.fit_transform(data_joined)

    pca_docs = pca_projection[:docs.shape[0]][rndperm][:n_pca_samples]
    pca_code = pca_projection[docs.shape[0]:][rndperm][:n_pca_samples]

    pca_joined = np.concatenate((pca_docs, pca_code), axis=0)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=tsne_iterations)
    tsne_projection = tsne.fit_transform(pca_joined)

    tsne_docs = tsne_projection[:n_tsne_samples]
    tsne_code = tsne_projection[n_pca_samples:n_pca_samples + n_tsne_samples]

    return tsne_docs, tsne_code

    #plt.plot([tsne_docs[:, 0], tsne_code[:, 0]], [tsne_docs[:, 1], tsne_code[:, 1]], color="k", zorder=1)
    #plt.scatter(tsne_docs[:, 0], tsne_docs[:, 1], zorder=2, color="orange")
    #plt.scatter(tsne_code[:, 0], tsne_code[:, 1], zorder=2, color="green")

    #docs_legend = mpatches.Patch(color="orange", label="Docs")
    #code_legend = mpatches.Patch(color="green", label="Code")
    #plt.legend(handles=[docs_legend, code_legend])
    #plt.show()

def execute(args):

    dataloader=None

    try:
        docs_untrained = torch.load(f"cache/untrained_ruby_docs_train.pt")
        code_untrained = torch.load(f"cache/untrained_ruby_code_train.pt")
        print(f"Untrained embeddings loaded")

    except:
        model = xMoCoVanillaModelPTL(args)

        dataloader = generateDataLoader("ruby", "train", model.docs_tokenizer, model.code_tokenizer, args)

        docs_untrained, code_untrained = generateEmbeddings(model, dataloader, "untrained_train")
        docs_untrained=docs_untrained.half()
        code_untrained=code_untrained.half()
        torch.save(docs_untrained, f"cache/untrained_ruby_docs_train.pt")
        torch.save(code_untrained, f"cache/untrained_ruby_code_train.pt")
        print(f"Untrained embeddings generated")

    try:
        docs_trained = torch.load(f"cache/ruby_docs_train.pt")
        code_trained = torch.load(f"cache/ruby_train.pt")
        print(f"Trained embeddings loaded")
    except:
        model = xMoCoModelPTL.load_from_checkpoint(f"checkpoints/ruby.ckpt")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

        if dataloader==None:
            dataloader = generateDataLoader("ruby", "train", tokenizer, tokenizer, args)

        docs_untrained, code_untrained = generateEmbeddings(model, dataloader, "trained_train")
        docs_untrained = docs_untrained.half()
        code_untrained = code_untrained.half()
        torch.save(docs_untrained, f"cache/ruby_docs_train.pt")
        torch.save(code_untrained, f"cache/ruby_train.pt")
        print(f"Trained embeddings generated")

    print(f"Everything preprocessed or loaded successfully!")

    #print(f"Now we could do dimensionality reduction, but it's not implemented yet...")

    docs_untrained = docs_untrained.cpu().numpy()
    code_untrained = code_untrained.cpu().numpy()
    docs_trained = docs_trained.cpu().numpy()
    code_trained = code_trained.cpu().numpy()

    # starting from here, the code is heavily inspired by https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    tsne_docs_untrained, tsne_code_untrained = downprojection(docs_untrained, code_untrained)
    tsne_docs_trained, tsne_code_trained = downprojection(docs_trained, code_trained)

    plt.subplot(1, 2, 1)

    plt.plot([tsne_docs_untrained[:, 0], tsne_code_untrained[:, 0]], [tsne_docs_untrained[:, 1], tsne_code_untrained[:, 1]], color="k", zorder=1)
    plt.scatter(tsne_docs_untrained[:, 0], tsne_docs_untrained[:, 1], zorder=2, color="orange")
    plt.scatter(tsne_code_untrained[:, 0], tsne_code_untrained[:, 1], zorder=2, color="green")

    docs_legend = mpatches.Patch(color="orange", label="Docs")
    code_legend = mpatches.Patch(color="green", label="Code")
    plt.legend(handles=[docs_legend, code_legend])
    plt.title("Pre-training embeddings")

    plt.subplot(1, 2, 2)

    plt.plot([tsne_docs_trained[:, 0], tsne_code_trained[:, 0]], [tsne_docs_trained[:, 1], tsne_code_trained[:, 1]], color="k", zorder=1)
    plt.scatter(tsne_docs_trained[:, 0], tsne_docs_trained[:, 1], zorder=2, color="orange")
    plt.scatter(tsne_code_trained[:, 0], tsne_code_trained[:, 1], zorder=2, color="green")

    docs_legend = mpatches.Patch(color="orange", label="Docs")
    code_legend = mpatches.Patch(color="green", label="Code")
    plt.legend(handles=[docs_legend, code_legend])
    plt.title("Post-training embeddings")

    plt.show()

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




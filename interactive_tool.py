# WIP: Basic shape works, next I'll have to add checkpoint loading and preprocessing (if uncached) and maybe a loading bar for preprocessing, if that is even possible
# for evaluating, make a new xMoCoModelPTL with stripped-down functionality and use the test_dataloader to load the corresponding subsets. note that we only have to forward the code, not the docstrings

from appJar import *
import torch
import torch.nn.functional as F
from xMoCo_pl import xMoCoModelPTL
from utils.multimodal_data_loading import generateDataLoader
from transformers import AutoTokenizer
import argparse
import sys
import json

model = None
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def loadJson(filename):
    cache=[]
    with open(filename, encoding="utf-8") as file:
        for line in file:
            line=line.strip()
            js = json.loads(line)
            cache.append(js)
    return cache

@torch.no_grad()
def generateEmbeddings(model, dataloader, subset):
    #forwards the given dataloader through the code_fast_encoder to obtain embeddings
    embedding_tensor=torch.tensor([]).cuda()

    if model.code_fast_encoder.device==torch.device("cpu"): #if it's on cpu, put it on gpu
        model.code_fast_encoder.cuda()

    if model.code_fast_encoder.training:
        model.code_fast_encoder.eval()

    for i, batch in enumerate(dataloader):
        code_samples = {"input_ids": batch["code_input_ids"].cuda(), "attention_mask": batch["code_attention_mask"].cuda()}
        code_embeddings = model.code_fast_encoder(input_ids=code_samples["input_ids"], attention_mask=code_samples["attention_mask"])["pooler_output"]
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)
        embedding_tensor = torch.cat((embedding_tensor, code_embeddings), dim=0)

        sys.stdout.write(f"\rForwarding code samples ({subset} subset): {i+1} / {len(dataloader)} ({(i+1)/len(dataloader)*100:.2f}%)")
        sys.stdout.flush()

    return embedding_tensor

def getCodeSamples(button):
    global model, tokenizer

    #need to reload these every time because the language might have changed
    train_json=None
    val_json = None
    test_json = None

    if button=="OK":
        app.clearTextArea("outputs")
        #app.setTextArea("outputs", "this is \n a text \n\n by me\t and \t him")
        query = app.getEntry("Query: ")
        language = app.getOptionBox("Language: ").lower()
        num_samples = app.getScale("Number of outputs: ")
        subsets = app.getProperties("Subsets to search in: ")

        if subsets["Train"]:
            train_json=loadJson(args.base_data_folder+f"/{language}/train.jsonl")
        if subsets["Validation"]:
            val_json=loadJson(args.base_data_folder+f"/{language}/valid.jsonl")
        if subsets["Test"]:
            test_json=loadJson(args.base_data_folder+f"/{language}/test.jsonl")

        print(query, language, num_samples, subsets)

        # [CHECK IF EMBEDDINGS ARE CACHED]:
        code_embeddings = {"train": torch.tensor([]).cuda(), "valid": torch.tensor([]).cuda(), "test": torch.tensor([]).cuda()}
        for key in subsets:
            if subsets[key]:
                key = key.lower()
                if key == "validation":
                    key = "valid"
                try:
                    code_embeddings[key] = torch.load(f"cache/{language}_{key}.pt")
                except:
                    print(f"No cached tensors found. Generating embeddings...")
                    if model==None:
                        model=xMoCoModelPTL.load_from_checkpoint(f"checkpoints/{language}.ckpt")

                    dataloader=generateDataLoader(language, key, tokenizer, tokenizer, args)
                    code_embeddings[key]=generateEmbeddings(model, dataloader, key)
                    torch.save(code_embeddings[key], f"cache/{language}_{key}.pt")

        assert code_embeddings["train"].shape[0]>0 or code_embeddings["valid"].shape[0]>0 or code_embeddings["test"].shape[0]>0, "Please select at least one subset to search in"

        embeddings_matrix = torch.cat((code_embeddings["train"], code_embeddings["valid"], code_embeddings["test"]))

        if model == None:
            model = xMoCoModelPTL.load_from_checkpoint(f"checkpoints/{language}.ckpt")

        # tokenize and encode the given docstring
        query_tokenized = tokenizer(query, truncation=True, padding="max_length")

        input_tokens=torch.tensor([query_tokenized["input_ids"]]).cuda()
        input_mask=torch.tensor([query_tokenized["attention_mask"]]).cuda()

        if model.docs_fast_encoder.device==torch.device("cpu"):
            model.docs_fast_encoder.cuda()

        if model.docs_fast_encoder.training:
            model.docs_fast_encoder.eval()
        with torch.no_grad():
            query_embedding = model.docs_fast_encoder(input_ids=input_tokens, attention_mask=input_mask)["pooler_output"]
            query_embedding = F.normalize(query_embedding, p=2, dim=1)

        similarity_matrix = torch.matmul(query_embedding, embeddings_matrix.T)
        top_similarities, top_indices = torch.topk(similarity_matrix, num_samples)

        after_str = "#############################################################################\n\n"
        output_str = after_str

        train_size = code_embeddings["train"].shape[0]
        val_size = code_embeddings["valid"].shape[0]
        test_size = code_embeddings["test"].shape[0]
        for i, code_index in enumerate(top_indices[0].tolist()):
            if code_index < code_embeddings["train"].shape[0]:
                data=train_json[code_index]
            elif code_index < train_size+val_size:
                data=val_json[code_index-train_size]
            elif code_index<train_size+val_size+test_size:
                data=test_json[code_index-train_size-val_size]
            else:
                assert False, "Something went wrong..."
            #confidence=torch.exp(top_similarities[0][i])/torch.sum(torch.exp(similarity_matrix))
            header_str = f"{i+1}: (similarity: {top_similarities[0][i].item():.4f})\n{data['url']}\n\n"
            output_str += header_str + data["code"] +"\n\n" + after_str

        app.clearTextArea("outputs")
        app.setTextArea("outputs", output_str)


        #st="def save_act(self, path=None):\n        \"\"\"Save model to a pickle located at `path`\"\"\"\n        if path is None:\n            path = os.path.join(logger.get_dir(), \"model.pkl\")\n\n        with tempfile.TemporaryDirectory() as td:\n            save_variables(os.path.join(td, \"model\"))\n            arc_name = os.path.join(td, \"packed.zip\")\n            with zipfile.ZipFile(arc_name, 'w') as zipf:\n                for root, dirs, files in os.walk(td):\n                    for fname in files:\n                        file_path = os.path.join(root, fname)\n                        if file_path != arc_name:\n                            zipf.write(file_path, os.path.relpath(file_path, td))\n            with open(arc_name, \"rb\") as f:\n                model_data = f.read()\n        with open(path, \"wb\") as f:\n            cloudpickle.dump((model_data, self._act_params), f)"
        #app.setTextArea("outputs", "1: (similarity: xxx, match probability: xxx, repo path: xxx)\n\n"+st+"\n\n"+after_str)

parser = argparse.ArgumentParser()
parser.add_argument("--base_data_folder", type=str, default="datasets/CodeSearchNet")
parser.add_argument("--effective_batch_size", type=int, default=8)
parser.add_argument("--debug_data_skip_interval", type=int, default=1)
parser.add_argument("--always_use_full_val", default=True)
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
parser.add_argument("--num_hard_negatives", type=int, default=0)
args = parser.parse_args()

app=gui("CodeSearch Tool", "1600x900")
app.setPadding([5, 5])
app.setSticky("ew")
app.setFont(14)

app.addLabelEntry("Query: ", 0, 0, 1)

app.addButton("OK", getCodeSamples, 0, 1, 1)
app.addLabelOptionBox("Language: ", ["Ruby", "JavaScript", "Java", "Go", "PHP", "Python"], 0, 2, 1)

subset = {"Train": True, "Validation": True, "Test": True}
app.addProperties("Subsets to search in: ", subset, 1, 2, 1)

app.addLabelScale("Number of outputs: ", 2, 2, 1)
app.setScaleRange("Number of outputs: ", 1, 20)
app.showScaleIntervals("Number of outputs: ", 19)
app.showScaleValue("Number of outputs: ")

app.addScrolledTextArea("outputs", 1, 0, 2, 2)

app.go()

app.go()
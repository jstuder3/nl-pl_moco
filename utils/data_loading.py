import torch
from datasets import load_dataset
import sys
import re

from utils.eda import eda as eda
from utils.codeSearchNetDataset import CodeSearchNetDataset

# takes as input a dict that has a key "func_documentation_string", shortens that to the first paragraph and puts the result under the key "func_documentation_string_shortened"
def shorten_data(dict):
    shortened_doc = " ".join(dict["func_documentation_string"].partition("\n\n")[0].split())  # shortens to the first paragraph and removes all "\n" and multi-whitespaces from the remainders
    dict["func_documentation_string_shortened"] = shortened_doc
    return dict

# takes as input a dict that has a key "whole_func_string", filters it for comments and shortens it a bit
# and returns the dict with an added key that contains the processed "whole_func_string"
def joinCodeTokens(dict, useCustomCodePreprocessing=False, useImprovedCodeConcatenation=False):
    if useCustomCodePreprocessing: # MAY HURT PERFORMANCE!
        func_string_cache=re.sub("((\n+)|(\r\n)+)", "\n", dict["whole_func_string"]) # turn all windows-like line terminators into unix-like line terminators (absolutely necessary because those sometimes cause non-termination for the next line somehow), also remove multi-linebreaks
        func_string_cache=re.sub("\"\"\"(.|\n)+\"\"\"", "",  func_string_cache, count=1) #remove docstrings, make sure to not waste time by only removing first occurence
        func_string_cache=re.sub(" {4}", "\t", func_string_cache) #replace every set of 4 multi-spaces with a tab
        dict["func_code_string_cleaned"]=func_string_cache
    else:
        if useImprovedCodeConcatenation:  # shorten the string a bit by removing unnecessary whitespaces # MAY ALSO HURT PERFORMANCE!
            for index in range(len(dict["func_code_tokens"])): #add line break after comments
                if len(dict["func_code_tokens"][index])>0 and dict["func_code_tokens"][index][0]=="#":
                    dict["func_code_tokens"][index]+="\n"
            concatenated_tokens = " ".join(dict["func_code_tokens"])
            pattern_and_target=[[" ( ", "("],
                                [" ) ", ") "],
                                [" [ ", "["],
                                [" ] ", "] "],
                                [" : ", ": "],
                                [" = ", "="],
                                [" == ", "=="],
                                [" != ", "!="],
                                [" <= ", "<="],
                                [" >= ", ">="],
                                [" < ", "<"],
                                [" > ", ">"],
                                [" , ", ", "],
                                [" . ", "."],
                                [" )", ")"],
                                [" ]", "]"],
                                [" :", ":"]]
            for pattern in pattern_and_target:
                concatenated_tokens=concatenated_tokens.replace(pattern[0], pattern[1])
        else: # SIMPLEST VERSION THAT SEEMS TO PROVIDE THE BEST PERFORMANCE
            concatenated_tokens = " ".join(dict["func_code_tokens"])
        dict["func_code_string_cleaned"] = concatenated_tokens

    return dict

def loadAndPreprocessData(source, language, split):
    dataset = load_dataset(source, language, split=split)

    dataset = dataset.map(shorten_data)
    dataset = dataset.map(joinCodeTokens)

    return dataset

# takes in a set of preprocessed (shortened) data, then augments it if augment is set to true, tokenizes it using
# the provided tokenizer and turns it into a CodeSearchNetDataset object and finally puts it into
# a DataLoader object which is returned
def generateDataLoader(source, language, split, tokenizer, batch_size, shuffle=False, augment=False):
    # we may want to augment several times independently and reloading the original data every time
    # is the only way I could find to make sure we start from the original data every time (Datasets have no copy method)
    preprocessed_data = loadAndPreprocessData(source, language, split)
    if augment:
        for i in range(len(preprocessed_data)):
            if i % 50 == 0:
                sys.stdout.write(f"\rAugmentation process: {i}/{len(preprocessed_data)} ({i/len(preprocessed_data)*100:.1f}%)")
                sys.stdout.flush()
            # augmentation for NL
            # if (re.search('[a-zA-Z]', preprocessed_data[i]["func_documentation_string_shortened"])): # necessary because there's some docstrings that are exclusively in different languages/non-latin alphabets and that breaks the eda code
            try:  # can throw errors in some very rare cases that I was not able to debug because it only occured on the server
                if (len(eda.get_only_chars(preprocessed_data[i]["func_documentation_string_shortened"])) > 0):
                    docs_augmentation_list = eda.eda(preprocessed_data[i]["func_documentation_string_shortened"],
                                                     num_aug=1)  # use default alphas for now
                    preprocessed_data[i]["func_documentation_string_shortened"] = docs_augmentation_list[0]
            except:
                print("Encountered error during augmentation. Sentence: " + str(
                    preprocessed_data[i]["func_documentation_string_shortened"]))
            # if docs_augmentation_list[0]!=augmentation_list[1]:
            #    print(f"Original:  {docs_augmentation_list[1]}\nAugmented: {docs_augmentation_list[0]}")

            # augmentation for code
            # for code we don't want synonym replacement or random insertion, so set the respective alphas to 0
            # unfortunately, this removes all non-alphanumerical characters, which is unsuitable for this task
            # code_augmentation_list = eda.eda(preprocessed_data[i]["func_code_string_cleaned"], alpha_sr=0, alpha_ri=0, num_aug=1)
            # preprocessed_data[i]["func_code_string_cleaned"]=code_augmentation_list[0]
            # if code_augmentation_list[0]!=code_augmentation_list[1]:
            #    print(f"Original:  {code_augmentation_list[1]}\nAugmented: {code_augmentation_list[0]}")

        # print once more to ensure next output starts on new line
        sys.stdout.write(f"\rAugmentation process: DONE (augmented {len(preprocessed_data)} samples)\n")
        sys.stdout.flush()
    docs_tokens = tokenizer(preprocessed_data["func_documentation_string_shortened"], truncation=True,
                            padding="max_length")
    code_tokens = tokenizer(preprocessed_data["func_code_string_cleaned"], truncation=True, padding="max_length")
    generated_dataset = CodeSearchNetDataset(docs_tokens, code_tokens)
    generated_loader = torch.utils.data.DataLoader(generated_dataset, batch_size=batch_size, shuffle=shuffle)
    return generated_loader
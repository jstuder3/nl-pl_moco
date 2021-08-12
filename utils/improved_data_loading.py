import torch
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
import sys
import re
import logging
import json

from utils.eda import eda as eda
from utils.codeSearchNetDataset import CodeSearchNetDataset

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

#COPIED FROM https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/run.py#L49
class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

#COPIED FROM https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/run.py#L60
def read_examples(filename, args):
    """Read examples from filename."""
    examples=[]
    counter=0
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):

            if idx%args.debug_data_skip_interval==0:
                if counter%100==0:
                    sys.stdout.write(f"\rData loading process: {idx}")
                    sys.stdout.flush()
                counter += 1

                line=line.strip()
                js=json.loads(line)
                if 'idx' not in js:
                    js['idx']=idx
                code=' '.join(js['code_tokens']).replace('\n',' ')
                code=' '.join(code.strip().split())
                nl=' '.join(js['docstring_tokens']).replace('\n','')
                nl=' '.join(nl.strip().split())
                examples.append(
                    Example(
                            idx = idx,
                            source=code,
                            target = nl,
                            )
                )
    sys.stdout.write(f"\rData loading process: DONE (loaded {counter} samples)\n")
    sys.stdout.flush()
    return examples

# COPIED FROM https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/run.py#L83

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask

# ADAPTED FROM https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/run.py#L101
def convert_examples_to_features(examples, tokenizer, stage=None):
    features = []
    counter=0
    for example_index, example in enumerate(examples):
        if counter % 100 == 0 and counter!=0:
            sys.stdout.write(f"\rTokenization process: {counter}/{len(examples)} ({counter/len(examples)*100:.1f}%)")
            sys.stdout.flush()
        counter+=1

        #this is fairly slow. maybe we can speed this up by not processing every sample on its own.
        # but for that we'd need to change the Example class and maybe turn it into a dictionary

        # code
        source_tokenized=tokenizer(example.source, truncation=True, padding="max_length")
        # nl
        target_tokenized=tokenizer(example.target, truncation=True, padding="max_length")

        if False and example_index < 3:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in tokenizer.decode(source_tokenized["input_ids"])]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_tokenized["input_ids"]))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_tokenized["attention_mask"]))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in tokenizer.decode(target_tokenized["input_ids"])]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_tokenized["input_ids"]))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_tokenized["attention_mask"]))))

        features.append(
            InputFeatures(
                example_index,
                source_tokenized["input_ids"],
                target_tokenized["input_ids"],
                source_tokenized["attention_mask"],
                target_tokenized["attention_mask"],
            )
        )
    sys.stdout.write(f"\rTokenization process: DONE (tokenized {counter} samples)\n")
    sys.stdout.flush()
    return features

# ADAPTED FROM https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/run.py
def generateDataLoader(language, split, tokenizer, args, shuffle=False, augment=False):
    # we may want to augment several times independently and reloading the original data every time
    # is the only way I could find to make sure we start from the original data every time (Datasets have no copy method)

    #load, preprocess, augment and tokenize data

    examples = read_examples(f"datasets/CodeSearchNet/{language}/{split}.jsonl", args)

    if augment:
        for i in range(len(examples)):
            if i % 100 == 0:
                sys.stdout.write(
                    f"\rAugmentation process: {i}/{len(examples)} ({i / len(examples) * 100:.1f}%)")
                sys.stdout.flush()
            # augmentation for NL
            try:
                docs_augmentation_list = eda.eda(examples[i].target, num_aug=1)  # use default alphas for now
                examples[i].target = docs_augmentation_list[0]
            except:
                print(f"\nEncountered error during augmentation. Problematic input: {examples[i].target}")

        # print once more to ensure next output starts on new line
        sys.stdout.write(f"\rAugmentation process: DONE (augmented {len(examples)} samples)\n")
        sys.stdout.flush()

    features = convert_examples_to_features(examples, tokenizer, stage=split)

    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)

    # source: code; target: nl
    data = CodeSearchNetDataset({"input_ids": all_target_ids, "attention_mask": all_target_mask}, {"input_ids": all_source_ids, "attention_mask": all_source_mask})

    dataloader = DataLoader(data, batch_size=args.batch_size, drop_last=True, shuffle=shuffle)

    return dataloader



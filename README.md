Repository for my bachelor thesis, where I try to improve on previous code search (NL to PL) methods by applying the MoCoV2 and xMoCo frameworks.

For comparable results, it is necessary to download the pre-processed and pre-filtered CodeSearchNet dataset by the authors of CodeBERT from https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/code2nl (download link: https://drive.google.com/file/d/1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h).

Place the unpacked data (the "CodeSearchNet" folder) in "datasets/" (so you would e.g. have the Python training data under "datasets/CodeSearchNet/python/train.jsonl")

To download this using command line, do the following:

    pip install gdown
    mkdir datasets
    cd datasets/
    gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
    unzip Cleaned_CodeSearchNet.zip
    rm Cleaned_CodeSearchNet.zip
    cd ..

There are many flags for main_MoCoV2_pl.py, some for hyperparameters and some for debugging:

    --model_name: huggingface model path (needs to have 768 embedding size)
    --num_epochs: number of epochs to train for
    --effective_batch_size: effective batch size. a batch of effective_batch_size/num_gpus is used per gpu
    --learning_rate: learning rate
    --temperature: temperature parameter
    --effective_queue_size: number of samples the queue can hold
    --momentum_update_weight: factor used for the momentum encoder updates
    --shuffle: enables shuffling of the training data
    --augment: enables EDA augmentation of the NL part of the training data
    --disable_normalizing_encoder_embeddings_during_training: if set, the output of the base model will not be normalized before being fed to the MLP 
    --disable_mlp: disables the MLP head and trains directly on the output of the base model
    --base_data_folder: folder that contains the cleaned CodeSearchNet dataset
    --debug_data_skip_interval: take only every i-th entry of the train/val dataset to decrease amount of data used
    --always_use_full_val: ignores the debug_data_skip_interval flag for the validation set
    --seed: seed used for randomized operations such as MLP initialization or EDA augmentation (needed to get correct results on multi-GPU training where every GPU has its own copy of the model)
    --accelerator: PyTorch Lightning accelerator to use
    --plugins: PyTorch Lightning plugins to use
    --precision: floating point precision to use
    --num_gpus: number of GPUs to use on multi-gpu machines

An example command can be seen below:

    python main_pl_new_ds.py --augment --num_epochs=20 --learning_rate=2e-5 --debug_data_skip_interval 1 --effective_queue_size=4096 --effective_batch_size=64 --num_gpus=4 --base_data_folder="/itet-stor/jstuder/net_scratch/nl-pl_moco/datasets/CodeSearchNet" --accelerator="ddp"
    
For xMoCo_pl.py, there's some additional flags:

    --docs_encoder: model to use for processing of the NL strings
    --code_encoder: model to use for processing of the PL strings
    --language: selects dataset on which should be trained/evaluated


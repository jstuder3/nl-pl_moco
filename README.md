Repository for bachelor thesis, where I try to improve on previous nl-pl methods by applying the MoCoV2 (and later possibly the xMoCo) framework.

For comparable results (using main_new_ds.py or main_pl_new_ds.py), it is required to download the pre-processed and pre-filtered CodeSearchNet dataset by the authors of CodeBERT from https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/code2nl 
(download link: https://drive.google.com/file/d/1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h).

Place the unpacked data (the "CodeSearchNet" folder) in "datasets/" (so you would e.g. have the Python training data under "datasets/CodeSearchNet/python/train.jsonl")

To download this using command line, do the following:

    pip install gdown
    mkdir datasets
    cd datasets/
    gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
    unzip Cleaned_CodeSearchNet.zip
    rm Cleaned_CodeSearchNet.zip
    cd ..

There are many flags, some for hyperparameters and some for debugging:

    --model_name: huggingface model path (needs 512 input and 768 embedding size)
    --num_epochs: number of epochs to train for
    --batch_size: batch size per GPU
    --learning_rate: learning rate
    --temperature: temperature parameter
    --max_queue_size: number of mega-batches the queue can hold (one mega-batch contains num_gpus * batch_size samples)
    --momentum_update_weight: factor used for the momentum encoder updates
    --shuffle: enables shuffling of the training data
    --augment: enables EDA augmentation of the NL part of the training data
    --normalize_encoder_embeddings_during_training: if enabled, the output of the base model will be normalized before being fed to the MLP
    --disable_mlp: disables the MLP head and trains directly on the output of the base model
    --base_data_folder: folder that contains the cleaned CodeSearchNet dataset
    --debug_data_skip_interval: take only every i-th entry of the train/val dataset to decrease amount of data used
    --always_use_full_val: ignores the debug_data_skip_interval flag for the validation set
    --seed: seed used for randomized operations such as MLP initialization or augmentation (needed to get correct results on multi-GPU training where every GPU has its own copy of the model)
    --accelerator: PyTorch Lightning accelerator to use
    --plugins: PyTorch Lightning plugins to use
    --precision: floating point precision to use

An example command can be seen below:

    python main_pl_new_ds.py --augment --shuffle --debug_data_skip_interval=1 --max_queue_size=128 --batch_size=16 --base_data_folder="/itet-stor/jstuder/net_scratch/nl-pl_moco/datasets/CodeSearchNet" --accelerator="ddp"
    
Note: For correct results, it is necessary that max_queue_size < len(train_loader) / num_gpus. For example, if you use --debug_data_skip_interval=1, we use the full 252k training samples for Python, so len(train_loader) = 252k / batch_size. Since the amount of data per queue entry varies depending on the batch_size and number of GPUs used, this effectively means that max_queue_size < 252k / (debug_data_skip_interval * batch_size * num_gpus) must hold.

# Tensorflow Implementation of DCN+
This repo contains a Tensorflow implementation of [Dynamic Coattention Network Plus](https://arxiv.org/abs/1711.00106).

# Environment Setup
Run `pip3 install -r requirements.txt`.

# Data Preprocessing 
See the `preprocessing` directory.

# Usage
In order to train a DCN+ model, simply run `python train.py`.

All of the relevant parameters for training are defined in `params.json` as follows:

`model_directory`: The directory where all of the graph and session information is saved

`model_repository`: The directory where copies of the `model_directory` are stored at the end of every epoch

`train_directory`: The directory that contains the training data

`test_directory`: The directory that contains the test data

`n_epochs`: The number of epochs that you want to run

`start_epoch`: The starting epoch (usually 1, but could be higher if you're resuming training for some reason)

`train_batch_size`: The batch size used for training

`test_batch_size`: The batch size used when computing predictions in testing

`state_size`: The state size for all RNNs in the model

`max_sequence_length`: The maximum sequence length for all RNNs in the model

`use_mixed_loss`: Flag to indicate whether to use the mixed loss. If false, just the cross-entropy loss is used.

`keep_prob`: The keep probability for dropout for all RNNs in the model

`word_keep_prob`: The keep probability for dropout as applied to words in the context document

`embedding_length`: The embedding length of the input data

`lr`: The learning rate of the optimizer

# Results
Using the training params in `params.json`, I was able to achieve 61.79% exact match score and 72.56% F1 score on the dev dataset. The model directory corresponding to this model is stored in this repo as `pretrained`. Make sure to clone the repo using `git lfs clone` if you want to use the pretrained model.

# Related Repos
[André Jonasson's implementation of DCN+](https://github.com/andrejonasson/dynamic-coattention-network-plus)

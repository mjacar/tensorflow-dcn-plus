# SQuAD Data Preprocessing
1. Download training data from https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json in this directory.
2. Download test data from https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json in this directory.
3. Download pre-trained GloVe word vectors from https://nlp.stanford.edu/projects/glove/.
4. Run `python data_writer.py --train_dir PATH_TO_TRAIN_DIRECTORY --test_dir PATH_TO_TEST_DIRECTORY --glove_file PATH_TO_GLOVE_FILE`. Additionally, you can use the optional flag `--use_cove` to use [CoVe](https://arxiv.org/abs/1708.00107). Note that using CoVE requires installation of [PyTorch](http://pytorch.org/), which is not included as part of the `requirements.txt` file.

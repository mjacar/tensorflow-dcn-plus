import json
import subprocess
import shutil
import tensorflow as tf
from model import dcn_plus_model
from data_reader import train_input_fn, predict_input_fn
from utils import evaluate
from nltk.tokenize.moses import MosesDetokenizer

if __name__ == '__main__':
  params = json.load(open('params.json'))
  detokenizer = MosesDetokenizer()
  tf.logging.set_verbosity(tf.logging.INFO)

  n_epochs = params['n_epochs']
  start_epoch = params['start_epoch']
  for epoch in range(start_epoch-1, n_epochs):
    dcn_estimator = tf.estimator.Estimator(
        model_fn=dcn_plus_model, params=params['model'], model_dir=params['model_directory'])
    dcn_estimator.train(input_fn=lambda: train_input_fn(
        params['train_directory'], params['train_batch_size'], params['model']['max_sequence_length'], params['model']['embedding_length']))
    predictions = dcn_estimator.predict(input_fn=lambda: predict_input_fn(
        params['test_directory'], params['test_batch_size'], params['model']['max_sequence_length'], params['model']['embedding_length']))
    predictions = list(predictions)
    answers = {}
    for prediction in predictions:
      answers[prediction['id'][0].decode('utf-8')] = detokenizer.detokenize([token.decode('utf-8')
                                                                             for token in prediction['context_tokens'][prediction['start']:prediction['end']+1]], return_str=True)
    exact_match, f1 = evaluate('preprocessing/dev-v1.1.json', answers)
    print("^^^^^^^^^^^^^^^^^^^^ Exact match: {} ^^^^^^^^^^^^^^^^^^^^".format(exact_match))
    print("^^^^^^^^^^^^^^^^^^^^ F1 score: {} ^^^^^^^^^^^^^^^^^^^^".format(f1))
    shutil.copytree(params['model_directory'],
                    '{}/epoch_{}_em_{:.2f}_f1_{:.2f}'.format(params['model_repository'], epoch+1, exact_match, f1))

import tensorflow as tf
from tensorflow.python.framework import function
from collections import Counter
from nltk.tokenize.moses import MosesDetokenizer
import string
import re
import json
import sys


def mask_to_start(score, start, score_mask_value=-1e30):
  """
  Mask score up to start
  Used for prediction
  """
  score_mask = tf.sequence_mask(start, maxlen=tf.shape(score)[1])
  score_mask_values = score_mask_value * tf.ones_like(score)
  return tf.where(~score_mask, score, score_mask_values)


def top_k_gpu(x, k):
  """
  Slightly modified version of _my_top_k function from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py
  Modified to deal with N-dimensional x tensor instead of 2-dimensional x tensor

  GPU-compatible version of top-k that works for very small constant k.
  Calls argmax repeatedly.
  tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
  seems not to be, so if we use tf.nn.top_k, then both the top_k and its
  gradient go on cpu.  Once this is not an issue, this function becomes
  obselete and should be replaced by tf.nn.top_k.
  """
  if k > 10:
    return tf.nn.top_k(x, k)
  values = []
  indices = []
  depth = tf.shape(x)[-1]
  for i in range(k):
    values.append(tf.reduce_max(x, -1))
    argmax = tf.argmax(x, -1)
    indices.append(argmax)
    if i + 1 < k:
      x += tf.one_hot(argmax, depth, float('-inf'))
  return tf.stack(values, axis=-1), tf.to_int32(tf.stack(indices, axis=-1))


def normalize_answer(s):
  """
  Lower text and remove punctuation, articles and extra whitespace.
  """
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, decode_bytes=False):
  """
  From the official evaluation script for the SQuAD dataset
  """
  if decode_bytes:
    prediction = prediction.decode('utf-8')
    ground_truth = ground_truth.decode('utf-8')
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0.0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def exact_match_score(prediction, ground_truth):
  """
  From the official evaluation script for the SQuAD dataset
  """
  return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
  """
  From the official evaluation script for the SQuAD dataset
  """
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths)


def evaluate(data_file, predictions):
  """
  From the official evaluation script for the SQuAD dataset
  """
  with open(data_file) as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']
  f1 = exact_match = total = 0
  for article in dataset:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        total += 1
        if qa['id'] not in predictions:
          message = 'Unanswered question ' + qa['id'] + \
                    ' will receive score 0.'
          print(message, file=sys.stderr)
          continue
        ground_truths = list(map(lambda x: x['text'], qa['answers']))
        prediction = predictions[qa['id']]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)

  exact_match = 100.0 * exact_match / total
  f1 = 100.0 * f1 / total

  return exact_match, f1


def detokenize(tokens, start, end):
  """
  Given a list of tokens, take the tokens from index start to index end and detokenize them
  """
  if end < start:
    return ''
  else:
    tokens = tokens[start:end+1]
    detokenizer = MosesDetokenizer()
    return detokenizer.detokenize([token.decode('utf-8') for token in tokens], return_str=True)


def tf_f1_score(tensors):
  """
  Given a predicted start, predicted end, actual start, actual end and a list of tokens, compute the F1 score
  """
  predicted_start = tensors[0][0]
  predicted_end = tensors[1][0]
  start = tensors[2][0]
  end = tensors[3][0]
  tokens = tensors[4]
  ground_truth = tf.py_func(
      detokenize, [tokens, start, end], tf.string, stateful=False)
  prediction = tf.py_func(
      detokenize, [tokens, predicted_start, predicted_end], tf.string, stateful=False)
  return tf.cast(tf.py_func(f1_score, [prediction, ground_truth, tf.constant(True)], tf.float64, stateful=False), tf.float32)

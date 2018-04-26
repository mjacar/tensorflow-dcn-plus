import argparse
import json
import os
import nltk
import torch
import numpy as np
import tensorflow as tf
from model import dcn_plus_model
from nltk.tokenize.moses import MosesDetokenizer
from preprocessing.cove_encoder import MTLSTM as CoveEncoder

def load_glove(filename):
  vocab_dict = {}
  embedding = []
  file = open(filename, 'r')
  for id, line in enumerate(file.readlines()):
    row = line.strip().split(' ')
    if len(row) != 301:
      continue
    vocab_dict[row[0]] = id
    embedding.append([float(i) for i in row[1:]])
  file.close()
  embedding.append([0] * len(embedding[0]))
  return vocab_dict, embedding


def get_vocab_id(word, vocab_dict):
  if vocab_dict.get(word) is None:
    return len(vocab_dict)
  else:
    return vocab_dict[word]


def pad_ids(id_list, vocab_dict, max_sequence_length):
  if len(id_list) >= max_sequence_length:
    return id_list[:max_sequence_length]
  else:
    return id_list + [len(vocab_dict)] * (max_sequence_length - len(id_list))


def pad_tokens(tokens, max_sequence_length):
  if len(tokens) >= max_sequence_length:
    return tokens[:max_sequence_length]
  else:
    pad_token = "<PAD>".encode('utf-8')
    return tokens + [pad_token] * (max_sequence_length - len(tokens))


def document_to_tensor(document, vocab_dict, embedding, max_sequence_length, cove_encoder):
  tokens = [token.replace("``", '"').replace(
      "''", '"') for token in nltk.word_tokenize(document)]
  length = [min(len(tokens), max_sequence_length)]
  tokens = pad_tokens(tokens, max_sequence_length)
  ids = pad_ids([get_vocab_id(token, vocab_dict)
                 for token in tokens], vocab_dict, max_sequence_length)
  tensor = [embedding[id] for id in ids]
  if cove_encoder is not None:
    inputs = torch.autograd.Variable(
        torch.LongTensor(np.asarray(ids))).unsqueeze(0).cuda()
    length = torch.LongTensor(np.asarray(length)).cuda()
    document_tensor, document_cove = cove_encoder(inputs, length)

    if document_cove.shape[1] < 600:
      document_cove = torch.cat([document_cove, torch.autograd.Variable(
          torch.zeros(1, 600 - document_cove.shape[1], 600)).cuda()], 1)
    document_tensor = torch.cat(
        [document_tensor, document_cove], 2).squeeze(0).data.cpu().numpy()

    for i in range(max_sequence_length):
      if ids[i] == len(vocab_dict):
        document_tensor[i] = np.zeros(900)
    tensor = document_tensor
  #document = tf.transpose(tf.constant(
  #    np.expand_dims(np.array(tensor), axis=0)), [0, 2, 1])
  document = tf.constant(np.expand_dims(np.array(tensor), axis=0))
  length = tf.constant(np.expand_dims(np.array(length), axis=0))
  return document, length


def input_fn(context, question, context_length, question_length, context_tokens):
  """
  features = [OrderedDict([('context', context), ('question', question), ('context_length', context_length),
              ('question_length', question_length), ('context_tokens', tf.constant(context_tokens)), ('id', tf.constant(np.array("id")))])]
  dtypes = OrderedDict([('context', tf.float32), ('question', tf.float32), ('context_length', tf.int64),
              ('question_length', tf.int64), ('context_tokens', tf.string), ('id', tf.string)])
  shapes = OrderedDict([('context', context.shape), ('question', question.shape), ('context_length', context_length.shape),
              ('question_length', question_length.shape), ('context_tokens', tf.constant(context_tokens).shape), ('id', tf.constant(np.array("id")).shape)])
  train_data = tf.data.Dataset.from_generator(lambda: (feature for feature in features), dtypes, shapes)
  """
  context_tokens = np.expand_dims(context_tokens, axis=0)
  features = {'context': context, 'question': question, 'context_length': context_length, 'question_length': question_length, 'context_tokens': tf.constant(context_tokens), 'id': tf.constant(np.array(["id"]))}
  train_data = tf.data.Dataset.from_tensors(features)
  iterator = train_data.make_one_shot_iterator()
  return iterator.get_next()


if __name__ == '__main__':
  params = json.load(open('params.json'))
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  max_sequence_length = params['model']['max_sequence_length']
  parser = argparse.ArgumentParser()
  parser.add_argument('--glove_file')
  parser.add_argument('--use_cove', action='store_true')
  parser.add_argument('--model_dir', nargs='?', default='pretrained', type=str)
  args = parser.parse_args()

  glove_file = args.glove_file
  if glove_file is None:
    print("Glove file needed")

  else:
    context = input("Context: ")
    question = input("Question: ")
    vocab_dict, embedding = load_glove(glove_file)

    cove_encoder = None
    if args.use_cove:
      cove_encoder = CoveEncoder(n_vocab=len(
          embedding), vectors=torch.FloatTensor(embedding), residual_embeddings=False)
      cove_encoder.cuda()

    context_tokens = [token.replace("``", '"').replace(
        "''", '"') for token in nltk.word_tokenize(context)]
    context_embedding, context_length = document_to_tensor(
        context, vocab_dict, embedding, max_sequence_length, cove_encoder)
    question_embedding, question_length = document_to_tensor(
        question, vocab_dict, embedding, max_sequence_length, cove_encoder)
    dcn_estimator = tf.estimator.Estimator(
        model_fn=dcn_plus_model, params=params['model'], model_dir=args.model_dir)
    prediction = dcn_estimator.predict(input_fn=lambda: input_fn(
        context_embedding, question_embedding, context_length, question_length, context_tokens))
    prediction = list(prediction)[0]
    detokenizer = nltk.tokenize.moses.MosesDetokenizer()
    prediction = detokenizer.detokenize([token.decode(
        'utf-8') for token in prediction['context_tokens'][prediction['start']:prediction['end']+1]], return_str=True)
    print("Answer: {}".format(prediction))

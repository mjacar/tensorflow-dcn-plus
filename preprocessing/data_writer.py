import argparse
import json
import numpy as np
import nltk
import tensorflow as tf
import torch
from glob import glob
from cove_encoder import MTLSTM as CoveEncoder


class DataWriter:
  def __init__(self, data_file, embedding_file, output_directory, max_sequence_length=600, use_cove=False):
    self.max_sequence_length = max_sequence_length
    self.vocab_dict, self.embedding = self.load_glove(embedding_file)
    self.embedding.append([0] * len(self.embedding[0]))
    self.data_file = data_file
    self.output_directory = output_directory
    self.use_cove = use_cove
    if use_cove:
      self.cove_encoder = CoveEncoder(n_vocab=len(
          self.embedding), vectors=torch.FloatTensor(self.embedding), residual_embeddings=False)
      self.cove_encoder.cuda()

  def load_glove(self, filename):
    """
    Takes a GloVe embedding file as input and outputs a word-to-id dictionary as well as an embedding matrix
    """
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
    return vocab_dict, embedding

  def get_vocab_id(self, word):
    if self.vocab_dict.get(word) is None:
      return len(self.vocab_dict)
    else:
      return self.vocab_dict[word]

  def pad_ids(self, id_list):
    if len(id_list) >= self.max_sequence_length:
      return id_list[:self.max_sequence_length]
    else:
      return id_list + [len(self.vocab_dict)] * (self.max_sequence_length - len(id_list))

  def pad_tokens(self, tokens):
    if len(tokens) >= self.max_sequence_length:
      return tokens[:self.max_sequence_length]
    else:
      pad_token = "<PAD>".encode('utf-8')
      return tokens + [pad_token] * (self.max_sequence_length - len(tokens))

  def token_idx_map(self, context, context_tokens):
    acc = ''
    current_token_idx = 0
    token_map = dict()
    for char_idx, char in enumerate(context):
      if char != ' ':
        acc += char
        context_token = str(context_tokens[current_token_idx])
        if acc == context_token:
          syn_start = char_idx - len(acc) + 1
          token_map[syn_start] = current_token_idx
          acc = ''
          current_token_idx += 1
    return token_map

  def write_train_record(self, writer, context_ids, question_ids, answer_start, answer_end, context_tokens):
    """
    Write individual TFRecord
    """
    if answer_start[0] >= self.max_sequence_length or answer_end[0] >= self.max_sequence_length:
      return
    context_length = [min(len(context_ids), self.max_sequence_length)]
    question_length = [min(len(question_ids), self.max_sequence_length)]
    context_ids = self.pad_ids(context_ids)
    question_ids = self.pad_ids(question_ids)
    context_tokens = self.pad_tokens(context_tokens)

    if not self.use_cove:
      context_tensor = [self.embedding[id] for id in context_ids]
      questions_tensor = [self.embedding[id] for id in question_ids]
      flat_context_tensor = [
          item for sublist in context_tensor for item in sublist]
      flat_question_tensor = [
          item for sublist in questions_tensor for item in sublist]

    else:
      context_inputs = torch.autograd.Variable(
          torch.LongTensor(np.asarray(context_ids))).unsqueeze(0).cuda()
      context_lengths = torch.LongTensor(np.asarray(context_length)).cuda()
      context_tensor, context_cove = self.cove_encoder(
          context_inputs, context_lengths)
      if context_cove.shape[1] < 600:
        context_cove = torch.cat([context_cove, torch.autograd.Variable(
            torch.zeros(1, 600 - context_cove.shape[1], 600)).cuda()], 1)
      context_tensor = torch.cat(
          [context_tensor, context_cove], 2).squeeze(0).data.cpu().numpy()

      question_inputs = torch.autograd.Variable(
          torch.LongTensor(np.asarray(question_ids))).unsqueeze(0).cuda()
      question_lengths = torch.LongTensor(np.asarray(question_length)).cuda()
      question_tensor, question_cove = self.cove_encoder(
          question_inputs, question_lengths)
      if question_cove.shape[1] < 600:
        question_cove = torch.cat([question_cove, torch.autograd.Variable(
            torch.zeros(1, 600 - question_cove.shape[1], 600)).cuda()], 1)
      question_tensor = torch.cat(
          [question_tensor, question_cove], 2).squeeze(0).data.cpu().numpy()

      for i in range(self.max_sequence_length):
        if context_ids[i] == len(self.vocab_dict):
          context_tensor[i] = np.zeros(900)
        if question_ids[i] == len(self.vocab_dict):
          question_tensor[i] = np.zeros(900)

      flat_context_tensor = [
          item for sublist in context_tensor for item in sublist]
      flat_question_tensor = [
          item for sublist in question_tensor for item in sublist]

    example = tf.train.Example(features=tf.train.Features(feature={
        'context': tf.train.Feature(float_list=tf.train.FloatList(value=flat_context_tensor)),
        'question': tf.train.Feature(float_list=tf.train.FloatList(value=flat_question_tensor)),
        'answer_start': tf.train.Feature(int64_list=tf.train.Int64List(value=answer_start)),
        'answer_end': tf.train.Feature(int64_list=tf.train.Int64List(value=answer_end)),
        'context_tokens': tf.train.Feature(bytes_list=tf.train.BytesList(value=context_tokens)),
        'context_length': tf.train.Feature(int64_list=tf.train.Int64List(value=context_length)),
        'question_length': tf.train.Feature(int64_list=tf.train.Int64List(value=question_length)),
    }))

    writer.write(example.SerializeToString())

  def write_test_record(self, writer, context_ids, question_ids, context_tokens, answers, id):
    """
    Write individual TFRecord
    """
    context_length = [min(len(context_ids), self.max_sequence_length)]
    question_length = [min(len(question_ids), self.max_sequence_length)]
    context_ids = self.pad_ids(context_ids)
    question_ids = self.pad_ids(question_ids)
    context_tokens = self.pad_tokens(context_tokens)
    answers = [str(chr(0)).join(answers).encode('utf-8')]

    if not self.use_cove:
      context_tensor = [self.embedding[id] for id in context_ids]
      questions_tensor = [self.embedding[id] for id in question_ids]
      flat_context_tensor = [
          item for sublist in context_tensor for item in sublist]
      flat_question_tensor = [
          item for sublist in questions_tensor for item in sublist]

    else:
      context_inputs = torch.autograd.Variable(
          torch.LongTensor(np.asarray(context_ids))).unsqueeze(0).cuda()
      context_lengths = torch.LongTensor(np.asarray(context_length)).cuda()
      context_tensor, context_cove = self.cove_encoder(
          context_inputs, context_lengths)
      if context_cove.shape[1] < 600:
        context_cove = torch.cat([context_cove, torch.autograd.Variable(
            torch.zeros(1, 600 - context_cove.shape[1], 600)).cuda()], 1)
      context_tensor = torch.cat(
          [context_tensor, context_cove], 2).squeeze(0).data.cpu().numpy()

      question_inputs = torch.autograd.Variable(
          torch.LongTensor(np.asarray(question_ids))).unsqueeze(0).cuda()
      question_lengths = torch.LongTensor(np.asarray(question_length)).cuda()
      question_tensor, question_cove = self.cove_encoder(
          question_inputs, question_lengths)
      if question_cove.shape[1] < 600:
        question_cove = torch.cat([question_cove, torch.autograd.Variable(
            torch.zeros(1, 600 - question_cove.shape[1], 600)).cuda()], 1)
      question_tensor = torch.cat(
          [question_tensor, question_cove], 2).squeeze(0).data.cpu().numpy()

      for i in range(self.max_sequence_length):
        if context_ids[i] == len(self.vocab_dict):
          context_tensor[i] = np.zeros(900)
        if question_ids[i] == len(self.vocab_dict):
          question_tensor[i] = np.zeros(900)

      flat_context_tensor = [
          item for sublist in context_tensor for item in sublist]
      flat_question_tensor = [
          item for sublist in question_tensor for item in sublist]

    example = tf.train.Example(features=tf.train.Features(feature={
        'context': tf.train.Feature(float_list=tf.train.FloatList(value=flat_context_tensor)),
        'question': tf.train.Feature(float_list=tf.train.FloatList(value=flat_question_tensor)),
        'context_length': tf.train.Feature(int64_list=tf.train.Int64List(value=context_length)),
        'question_length': tf.train.Feature(int64_list=tf.train.Int64List(value=question_length)),
        'context_tokens': tf.train.Feature(bytes_list=tf.train.BytesList(value=context_tokens)),
        'ground_truths': tf.train.Feature(bytes_list=tf.train.BytesList(value=answers)),
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=id))
    }))

    writer.write(example.SerializeToString())

  def write_train_data(self):
    """
    Write data to a series of TFRecord files
    """
    if not glob("{}/*".format(self.output_directory)):
      data = json.loads(open(self.data_file).read())
      for title in data['data']:
        writer = tf.python_io.TFRecordWriter(
            "{}/{}.tfrecords".format(self.output_directory, title['title']))
        for question in title['paragraphs']:
          context_tokens = [token.replace("``", '"').replace(
              "''", '"') for token in nltk.word_tokenize(question['context'])]
          answer_map = self.token_idx_map(question['context'].replace(
              "``", '"').replace("''", '"'), context_tokens)
          context_ids = [self.get_vocab_id(word.lower())
                         for word in context_tokens]
          context_tokens = [word.lower().encode('utf-8') for word in context_tokens]
          for question_answer_pair in question['qas']:
            question_tokens = [token.lower() for token in nltk.word_tokenize(
                question_answer_pair['question'].replace("``", '"').replace("''", '"'))]
            question_ids = [self.get_vocab_id(word)
                            for word in question_tokens]
            for answer in question_answer_pair['answers']:
              last_word_answer = len([token.replace("``", '"').replace(
                  "''", '"') for token in nltk.word_tokenize(answer['text'])][-1])
              answer_length = len(answer['text'])
              try:
                answer_start = [answer_map[answer['answer_start']]]
                answer_end = [
                    answer_map[answer['answer_start'] + answer_length - last_word_answer]]
                self.write_train_record(
                    writer, context_ids, question_ids, answer_start, answer_end, context_tokens)
              except:
                continue
        writer.close()

  def write_test_data(self):
    """
    Write data to a series of TFRecord files
    """
    if not glob("{}/*".format(self.output_directory)):
      data = json.loads(open(self.data_file).read())
      for title in data['data']:
        writer = tf.python_io.TFRecordWriter(
            "{}/{}.tfrecords".format(self.output_directory, title['title']))
        for question in title['paragraphs']:
          context_tokens = [token.replace("``", '"').replace(
              "''", '"') for token in nltk.word_tokenize(question['context'])]
          answer_map = self.token_idx_map(question['context'].replace(
              "``", '"').replace("''", '"'), context_tokens)
          context_ids = [self.get_vocab_id(word.lower())
                         for word in context_tokens]
          context_tokens = [token.lower().encode('utf-8')
                            for token in context_tokens]
          for question_answer_pair in question['qas']:
            question_tokens = [token.lower() for token in nltk.word_tokenize(
                question_answer_pair['question'].replace("``", '"').replace("''", '"'))]
            question_ids = [self.get_vocab_id(word)
                            for word in question_tokens]
            answers = []
            for answer in question_answer_pair['answers']:
              id = [question_answer_pair['id'].encode('utf-8')]
              answers.append(answer['text'])
            self.write_test_record(
                writer, context_ids, question_ids, context_tokens, answers, id)
        writer.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--glove_file')
  parser.add_argument('--train_dir')
  parser.add_argument('--test_dir')
  parser.add_argument('--use_cove', action='store_true')
  args = parser.parse_args()

  glove_file = args.glove_file
  if glove_file is None:
    print("Glove file needed")

  else:
    train_dir = args.train_dir
    test_dir = args.test_dir
    if train_dir is not None:
      data_writer = DataWriter(
        'train-v1.1.json', glove_file, train_dir, use_cove=args.use_cove)
      data_writer.write_train_data()
    if test_dir is not None:
      data_writer = DataWriter(
        'dev-v1.1.json', glove_file, test_dir, use_cove=args.use_cove)
      data_writer.write_test_data()

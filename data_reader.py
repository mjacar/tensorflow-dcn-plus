import tensorflow as tf
from glob import glob


def train_input_fn(data_directory, batch_size, max_sequence_length, embedding_length):
  """
  Input function to use for the estimator in TRAIN mode
  """
  train_data = tf.data.TFRecordDataset(glob("{}/*".format(data_directory)))
  # Shuffle the data files
  train_data = train_data.shuffle(buffer_size=500)
  train_data = train_data.map(parse_train_tfrecord(
      max_sequence_length, embedding_length))
  # Shuffle the individual records
  train_data = train_data.shuffle(buffer_size=5000)
  train_data = train_data.batch(batch_size)
  iterator = train_data.make_one_shot_iterator()
  return iterator.get_next()


def predict_input_fn(data_directory, batch_size, max_sequence_length, embedding_length):
  """
  Input function to use for the estimator in PREDICT mode
  """
  predict_data = tf.data.TFRecordDataset(glob("{}/*".format(data_directory)))
  predict_data = predict_data.map(parse_predict_tfrecord(
      max_sequence_length, embedding_length))
  predict_data = predict_data.batch(batch_size)
  iterator = predict_data.make_one_shot_iterator()
  return iterator.get_next()


def parse_train_tfrecord(max_sequence_length, embedding_length):
  """
  Returns a method for parsing .tfrecords files for TRAIN mode
  """
  def parser(raw_tfrecord):
    tfrecord_features = tf.parse_single_example(raw_tfrecord,
                                                features={
                                                    'context': tf.VarLenFeature(tf.float32),
                                                    'question': tf.VarLenFeature(tf.float32),
                                                    'answer_start': tf.VarLenFeature(tf.int64),
                                                    'answer_end': tf.VarLenFeature(tf.int64),
                                                    'context_tokens': tf.VarLenFeature(tf.string),
                                                    'context_length': tf.VarLenFeature(tf.int64),
                                                    'question_length': tf.VarLenFeature(tf.int64),
                                                })
    context = tfrecord_features['context'].values
    context = tf.reshape(context, [max_sequence_length, embedding_length])
    question = tfrecord_features['question'].values
    question = tf.reshape(question, [max_sequence_length, embedding_length])

    context_tokens = tfrecord_features['context_tokens'].values
    answer_start = tf.reshape(tfrecord_features['answer_start'].values, [1])
    answer_end = tf.reshape(tfrecord_features['answer_end'].values, [1])
    context_length = tf.reshape(
        tfrecord_features['context_length'].values, [1])
    question_length = tf.reshape(
        tfrecord_features['question_length'].values, [1])

    features = {'context': context, 'question': question,
                'context_length': context_length, 'question_length': question_length}
    labels = {'answer_start': answer_start,
              'answer_end': answer_end, 'context_tokens': context_tokens}
    return features, labels
  return parser


def parse_predict_tfrecord(max_sequence_length, embedding_length):
  """
  Returns a method for parsing .tfrecords files for PREDICT mode
  """
  def parser(raw_tfrecord):
    tfrecord_features = tf.parse_single_example(raw_tfrecord,
                                                features={
                                                    'context': tf.VarLenFeature(tf.float32),
                                                    'question': tf.VarLenFeature(tf.float32),
                                                    'context_length': tf.VarLenFeature(tf.int64),
                                                    'question_length': tf.VarLenFeature(tf.int64),
                                                    'context_tokens': tf.VarLenFeature(tf.string),
                                                    'id': tf.VarLenFeature(tf.string)
                                                }, name='features')
    context = tfrecord_features['context'].values
    context = tf.reshape(context, [max_sequence_length, embedding_length])
    question = tfrecord_features['question'].values
    question = tf.reshape(question, [max_sequence_length, embedding_length])

    id = tfrecord_features['id'].values
    context_tokens = tfrecord_features['context_tokens'].values
    context_length = tf.reshape(
        tfrecord_features['context_length'].values, [1])
    question_length = tf.reshape(
        tfrecord_features['question_length'].values, [1])

    features = {'context': context, 'question': question, 'context_length': context_length,
                'question_length': question_length, 'context_tokens': context_tokens, 'id': id}
    return features
  return parser

import tensorflow as tf


def cell(state_size, keep_prob):
  """
  Definition of LSTM cell to be used
  """
  cell = tf.contrib.rnn.LSTMCell(num_units=state_size)
  return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob, dtype=tf.float32)


def concat_sentinel(matrix, sentinel_name):
  """
  Concatenate trainable sentinel vector to matrix
  """
  sentinel = tf.get_variable(sentinel_name, matrix.get_shape()[2], tf.float32)
  sentinel = tf.reshape(sentinel, (1, 1, -1))
  sentinel = tf.tile(sentinel, (tf.shape(matrix)[0], 1, 1))
  return tf.concat([sentinel, matrix], 1)


def bilstm_encoding(inputs, sequence_length, state_size, keep_prob, scope):
  """
  Encode inputs via bidirectional LSTM
  """
  outputs, _ = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell(state_size, keep_prob),
      cell_bw=cell(state_size, keep_prob),
      dtype=tf.float32,
      sequence_length=sequence_length,
      inputs=inputs,
      scope=scope)
  encoding = tf.concat(outputs, 2)
  return encoding


def mask_affinity(affinity, sequence_length, affinity_mask_value=float('-inf')):
  """
  Mask affinity by sequence_length
  Used in computation of coattention
  """
  score_mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(affinity)[1])
  score_mask = tf.tile(tf.expand_dims(score_mask, 2),
                       (1, 1, tf.shape(affinity)[2]))
  affinity_mask_values = affinity_mask_value * tf.ones_like(affinity)
  return tf.where(score_mask, affinity, affinity_mask_values)


def coattention(document_encoding, question_encoding, batch, state_size, keep_prob, apply_sentinel):
  """
  Compute coattention between document_encoding and question_encoding
  """
  document_length = batch['context_length']
  question_length = batch['question_length']
  if apply_sentinel:
    document_encoding = concat_sentinel(document_encoding, 'document_sentinel')
    question_encoding = concat_sentinel(question_encoding, 'question_sentinel')
    document_length += 1
    question_length += 1
  unmasked_affinity = tf.einsum('ndh,nqh->ndq', document_encoding,
                                question_encoding, name='unmasked_affinity')
  affinity = tf.identity(mask_affinity(
      unmasked_affinity, document_length), name='affinity')
  attention_q = tf.nn.softmax(affinity, axis=1, name='attention_q')
  unmasked_affinity_t = tf.transpose(
      unmasked_affinity, [0, 2, 1], name='unmasked_affinity_t')
  affinity_t = tf.identity(mask_affinity(
      unmasked_affinity_t, question_length), name='affinity_t')
  attention_d = tf.nn.softmax(affinity_t, axis=1, name='attention_d')
  summary_q = tf.einsum('ndh,ndq->nqh', document_encoding,
                        attention_q, name='summary_q')
  summary_d = tf.einsum('nqh,nqd->ndh', question_encoding,
                        attention_d, name='summary_d')
  if apply_sentinel:
    summary_d = summary_d[:, 1:, :]
    summary_q = summary_q[:, 1:, :]
    attention_q = attention_q[:, 1:, 1:]
  coattention_d = tf.einsum('nqh,nqd->ndh', summary_q,
                            attention_q, name='coattention_d')
  return summary_q, summary_d, coattention_d


def encode(batch, state_size, keep_prob, word_keep_prob, embedding_length):
  with tf.variable_scope('document') as scope:
    batch['context'] = tf.nn.dropout(batch['context'], word_keep_prob)
    document_encoding = bilstm_encoding(
        batch['context'], batch['context_length'], state_size, keep_prob, scope)
  with tf.variable_scope('question') as scope:
    question_encoding = bilstm_encoding(
        batch['question'], batch['question_length'], state_size, keep_prob, scope)
    question_encoding = tf.layers.dense(
        question_encoding, question_encoding.get_shape()[2], activation=tf.tanh)
  with tf.variable_scope('coattention_1') as scope:
    summary_q_1, summary_d_1, coattention_d_1 = coattention(
        document_encoding, question_encoding, batch, state_size, keep_prob, apply_sentinel=True)
  with tf.variable_scope('summary_d') as scope:
    summary_d_encoding = bilstm_encoding(
        summary_d_1, batch['context_length'], state_size, keep_prob, scope)
  with tf.variable_scope('summary_q') as scope:
    summary_q_encoding = bilstm_encoding(
        summary_q_1, batch['question_length'], state_size, keep_prob, scope)
  with tf.variable_scope('coattention_2') as scope:
    summary_q_2, summary_d_2, coattention_d_2 = coattention(
        summary_d_encoding, summary_q_encoding, batch, state_size, keep_prob, apply_sentinel=False)
  with tf.variable_scope('final_encode') as scope:
    document = tf.concat([document_encoding, summary_d_encoding,
                          summary_d_1, summary_d_2, coattention_d_1, coattention_d_2], 2)
    outputs = bilstm_encoding(
        document, batch['context_length'], state_size, keep_prob, scope)
  return outputs

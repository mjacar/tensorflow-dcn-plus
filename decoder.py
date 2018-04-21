import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _maybe_mask_score
from utils import top_k_gpu


def get_encoding_at_index(encoding, index, batch_size):
  return tf.gather_nd(encoding, tf.stack([tf.range(batch_size), index], axis=1))


def maxout_layer(inputs, output_size=200, pool_size=16):
  """
  Maxout layer
  See https://arxiv.org/pdf/1302.4389.pdf
  """
  pool = tf.layers.dense(inputs, output_size*pool_size)
  pool = tf.reshape(pool, (-1, tf.shape(inputs)[1], output_size, pool_size))
  output = tf.reduce_max(pool, -1)
  return output


def sparsify(matrix, k=2):
  """
  Sparsify matrix in accordance with equation 3 from https://arxiv.org/pdf/1701.06538.pdf
  """
  _, top_indices = top_k_gpu(matrix, k)
  output = tf.one_hot(top_indices, tf.shape(matrix)[-1], 1.0, 0.0)
  output = tf.multiply(tf.reduce_sum(output, axis=-2), matrix)
  output = tf.nn.softmax(
      output + -1e30 * tf.cast(tf.equal(output, 0), tf.float32))
  return output


def sparse_mixture_of_experts_layer(inputs, output_size=200, num_experts=16):
  """
  Sparse mixture of experts layer
  See https://arxiv.org/pdf/1701.06538.pdf
  """
  non_noise = tf.layers.dense(
      inputs, output_size*num_experts, use_bias=False, name='gating_non_noise')
  noise = tf.layers.dense(inputs, output_size*num_experts,
                          activation=tf.nn.softplus, use_bias=False, name='gating_noise')
  h = non_noise + \
      tf.random_normal((tf.shape(inputs)[0], tf.shape(
          inputs)[1], output_size*num_experts)) * noise
  h = tf.reshape(h, (-1, tf.shape(inputs)[1], output_size, num_experts))
  g = sparsify(h)
  e = tf.layers.dense(inputs, output_size*num_experts,
                      activation=tf.tanh, name='experts')
  e = tf.reshape(e, (-1, tf.shape(inputs)[1], output_size, num_experts))
  output = tf.reduce_sum(tf.multiply(g, e), -1)
  return output


def highway_maxout(encoding, hidden_state, start, end, context_length, batch_size, state_size):
  """
  Highway maxout network
  Defined in original DCN paper: https://arxiv.org/pdf/1611.01604.pdf
  """
  start_encoding = get_encoding_at_index(encoding, start, batch_size)
  end_encoding = get_encoding_at_index(encoding, end, batch_size)
  r = tf.layers.dense(tf.concat([hidden_state, start_encoding, end_encoding],
                                axis=1), use_bias=False, activation=tf.tanh, units=state_size)
  r = tf.expand_dims(r, 1)
  r = tf.tile(r, (1, tf.shape(encoding)[1], 1))
  layer_1 = sparse_mixture_of_experts_layer(tf.concat([encoding, r], axis=2))
  layer_2 = maxout_layer(layer_1)
  output = maxout_layer(tf.concat([layer_1, layer_2], axis=2), 1)
  logit = tf.squeeze(output, -1)
  return _maybe_mask_score(logit, context_length, float('-inf'))


def decoder_body(encoding, state, start, end, context_length, batch_size, state_size):
  """
  Execute single timestep of decoder network
  """
  maxlen = tf.shape(encoding)[1]
  with tf.variable_scope('start'):
    alpha = highway_maxout(encoding, state, start, end,
                           context_length, batch_size, state_size)
  with tf.variable_scope('end'):
    updated_start = tf.argmax(alpha, axis=1, output_type=tf.int32)
    beta = highway_maxout(encoding, state, updated_start,
                          end, context_length, batch_size, state_size)
  return tf.stack([alpha, beta], axis=2)


def decode(encoding, context_length, state_size, keep_prob):
  with tf.variable_scope('decoder_loop', reuse=tf.AUTO_REUSE):
    batch_size = tf.shape(encoding)[0]
    lstm_dec = tf.contrib.rnn.LSTMCell(num_units=state_size)
    lstm_dec = tf.contrib.rnn.DropoutWrapper(
        lstm_dec, output_keep_prob=keep_prob, dtype=tf.float32)

    start = tf.zeros((batch_size,), dtype=tf.int32)
    end = tf.to_int32(context_length - 1)
    answer = tf.stack([start, end], axis=1)
    state = lstm_dec.zero_state(batch_size, dtype=tf.float32)
    not_settled = tf.tile([True], (batch_size,))
    logits = tf.TensorArray(tf.float32, size=4, clear_after_read=False)

    for i in range(4):
      start_encoding = get_encoding_at_index(encoding, start, batch_size)
      end_encoding = get_encoding_at_index(encoding, end, batch_size)
      answer_encoding = tf.concat([start_encoding, end_encoding], axis=1)
      output, state = lstm_dec(answer_encoding, state)
      if i == 0:
        logit = decoder_body(encoding, output, start, end,
                             context_length, batch_size, state_size)
      else:
        prev_logit = logits.read(i-1)
        logit = tf.cond(tf.reduce_any(not_settled), lambda: decoder_body(
            encoding, output, start, end, context_length, batch_size, state_size), lambda: prev_logit)
      start = tf.argmax(logit[:, :, 0], axis=1, output_type=tf.int32)
      end = tf.argmax(logit[:, :, 1], axis=1, output_type=tf.int32)
      new_answer = tf.stack([start, end], axis=1)
      if i == 0:
        not_settled = tf.tile([True], (batch_size,))
      else:
        not_settled = tf.reduce_any(tf.not_equal(answer, new_answer), axis=1)
      not_settled = tf.reshape(not_settled, (batch_size,))
      answer = new_answer
      logits = logits.write(i, logit)

  return logits

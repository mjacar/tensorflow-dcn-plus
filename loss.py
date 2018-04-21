import tensorflow as tf
from utils import mask_to_start, tf_f1_score


def cross_entropy_loss(logits, answer_start, answer_end):
  """
  Cross entropy loss across all decoder timesteps
  """
  logits = logits.concat()
  answer_span = tf.concat(
      [answer_start, answer_end], axis=1)
  answer_repeated = tf.tile(answer_span, (4, 1))

  start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits[:, :, 0], labels=answer_repeated[:, 0])
  end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits[:, :, 1], labels=answer_repeated[:, 1])
  start_loss = tf.stack(tf.split(start_loss, 4), axis=1)
  end_loss = tf.stack(tf.split(end_loss, 4), axis=1)
  loss = tf.reduce_mean(tf.reduce_mean(
      start_loss + end_loss, axis=1), axis=0)
  return loss


def reward(guess_start, guess_end, answer_start, answer_end, baseline, tokens, logits_shape):
  """
  Reinforcement learning reward (i.e. F1 score) from sampling a trajectory of guesses across each decoder timestep
  """
  reward = tf.zeros(logits_shape, dtype=tf.float32)
  reward = [reward] * 4
  for t in range(4):
    f1_score = tf.map_fn(
        tf_f1_score, (guess_start[t], guess_end[t], answer_start, answer_end, tokens), dtype=tf.float32)
    normalized_reward = tf.stop_gradient(f1_score - baseline)
    reward[t] = normalized_reward
  return tf.stack(reward)


def surrogate_loss(logits, guess_start, guess_end, r):
  """
  The surrogate loss to be used for policy gradient updates
  """
  logits = logits.concat()
  guess_start = tf.reshape(guess_start, [-1])
  guess_end = tf.reshape(guess_end, [-1])
  r = tf.reshape(r, [-1])

  start_loss = r * \
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits[:, :, 0], labels=guess_start)
  end_loss = r * \
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits[:, :, 1], labels=guess_end)
  start_loss = tf.stack(tf.split(start_loss, 4), axis=1)
  end_loss = tf.stack(tf.split(end_loss, 4), axis=1)
  loss = tf.reduce_mean(tf.reduce_mean(
      start_loss + end_loss, axis=1), axis=0)
  return loss


def rl_loss(logits, answer_start, answer_end, tokens):
  """
  Reinforcement learning loss
  """
  final_logits = logits.read(3)
  final_start_logits = final_logits[:, :, 0]
  final_end_logits = final_logits[:, :, 1]
  guess_start_greedy = tf.argmax(final_start_logits, axis=1)
  end_guess_greedy = tf.argmax(mask_to_start(
      final_end_logits, guess_start_greedy), axis=1)
  guess_end_greedy = tf.argmax(final_end_logits, axis=1)
  guess_start_greedy = tf.expand_dims(guess_start_greedy, axis=1)
  guess_end_greedy = tf.expand_dims(guess_end_greedy, axis=1)
  baseline = tf.map_fn(tf_f1_score, (guess_start_greedy, guess_end_greedy,
                                     answer_start, answer_end, tokens), dtype=tf.float32)

  guess_start = []
  guess_end = []
  for t in range(4):
    logits_t = logits.read(t)
    start_logits = logits_t[:, :, 0]
    end_logits = logits_t[:, :, 1]
    guess_start.append(tf.multinomial(start_logits, 1))
    guess_end.append(tf.multinomial(end_logits, 1))
  guess_start = tf.stack(guess_start)
  guess_end = tf.stack(guess_end)

  r = reward(guess_start, guess_end, answer_start, answer_end,
             baseline, tokens, tf.shape(final_start_logits))
  surr_loss = surrogate_loss(logits, guess_start, guess_end, r)
  loss = tf.reduce_mean(-r)

  # This function needs to return the value of loss in the forward pass so that theta_rl gets the right parameter update
  # However, this needs to have the gradient of surr_loss in the backward pass so the model gets the right policy gradient update
  return surr_loss + tf.stop_gradient(loss - surr_loss)

import tensorflow as tf
from encoder import encode
from decoder import decode
from loss import cross_entropy_loss, rl_loss
from utils import mask_to_start, tf_f1_score


def dcn_plus_model(features, labels, mode, params):
  if mode == tf.estimator.ModeKeys.EVAL:
    # It's more convenient to implement eval using the PREDICT mode
    raise NotImplementedError

  keep_prob = params['keep_prob'] if mode == tf.estimator.ModeKeys.TRAIN else 1.0
  word_keep_prob = params['word_keep_prob'] if mode == tf.estimator.ModeKeys.TRAIN else 1.0
  features['context_length'] = tf.squeeze(
      features['context_length'], [1], name='context_length')
  features['question_length'] = tf.squeeze(
      features['question_length'], [1], name='question_length')

  with tf.variable_scope('prediction'):
    coattention = tf.identity(
        encode(features, params['state_size'], keep_prob, word_keep_prob, params['embedding_length']), name='coattention')
    logits = decode(
        coattention, features['context_length'], params['state_size'], keep_prob)

    final_logits = logits.read(3)
    start_preds = tf.argmax(
        final_logits[:, :, 0], axis=1, name='start_prediction')
    end_preds = tf.argmax(mask_to_start(
        final_logits[:, :, 1], start_preds), axis=1, name='end_prediction')

  if mode == tf.estimator.ModeKeys.PREDICT:
    prediction = {
        'start': start_preds,
        'end': end_preds,
        'context_tokens': features['context_tokens'],
        'id': features['id']}
    return tf.estimator.EstimatorSpec(mode, predictions=prediction)

  else:
    with tf.variable_scope('loss'):
      loss_ce = cross_entropy_loss(
          logits, labels['answer_start'], labels['answer_end'])
      loss_rl = rl_loss(
          logits, labels['answer_start'], labels['answer_end'], labels['context_tokens'])
      if params['use_mixed_loss']:
        theta_ce = tf.get_variable('theta_ce', (), tf.float32)
        theta_rl = tf.get_variable('theta_rl', (), tf.float32)
        loss = (1/(2*theta_ce*theta_ce))*loss_ce + (1/(2*theta_rl*theta_rl)) * \
            loss_rl + tf.log(theta_ce * theta_ce) + tf.log(theta_rl * theta_rl)
      else:
        loss = loss_ce

    avg_f1 = 100 * tf.reduce_mean(tf.map_fn(tf_f1_score, (tf.expand_dims(start_preds, axis=1), tf.expand_dims(
        end_preds, axis=1), labels['answer_start'], labels['answer_end'], labels['context_tokens']), dtype=tf.float32))
    logging_hook = tf.train.LoggingTensorHook(
        {"loss_ce": loss_ce, "loss_rl": loss_rl, "avg_f1": avg_f1}, every_n_iter=1)
    train_op = tf.train.AdamOptimizer(params['lr']).minimize(
        loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

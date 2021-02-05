import numpy as np
import tensorflow as tf
import time
import sys
import os
from time import time
import pickle


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class TopicDisQuant(object):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        initializer = tf.uniform_unit_scaling_initializer()
        e1 = tf.Variable(tf.eye(embedding_dim, name='embedding'), trainable=True)
        if num_embeddings > embedding_dim:
            e2 = tf.get_variable('embedding', [embedding_dim, num_embeddings - embedding_dim], initializer=initializer, trainable=True)
            e2 = tf.transpose(e2)
            self._E = tf.Variable(tf.concat([e1, e2], axis=0))
        else:
            self._E = e1

    def forward(self, inputs):
        input_shape = tf.shape(inputs)
        with tf.control_dependencies([tf.Assert(tf.equal(input_shape[-1], self.embedding_dim), [input_shape])]):
            flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                    - 2 * tf.matmul(flat_inputs, tf.transpose(self._E))
                    + tf.transpose(tf.reduce_sum(self._E ** 2, 1, keepdims=True)))

        encoding_indices = tf.argmax(- distances, 1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])

        quantized = self.quantize(encoding_indices)

        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + tf.stop_gradient(quantized - inputs)

        return {
                'quantize': quantized,
                'loss': loss,
                'encodings': encodings,
                'e_latent_loss': e_latent_loss,
                'q_latent_loss': q_latent_loss
            }

    def quantize(self, encoding_indices):
        return tf.nn.embedding_lookup(self._E, encoding_indices, validate_indices=False)


class NQTM(object):

    def __init__(self, config):
        self.config = config
        self.active_fct = config['active_fct']
        self.keep_prob = config['keep_prob']
        self.word_sample_size = config['word_sample_size']
        self.topic_num = config['topic_num']
        self.exclude_topt = 1
        self.select_topic_num = int(self.topic_num - 2)

        self.topic_dis_quant = TopicDisQuant(self.topic_num, self.topic_num, commitment_cost=config['commitment_cost'])

        self.init()

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

    def init(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.config['vocab_size']))
        self.w_omega = tf.placeholder(dtype=tf.float32, name='w_omega')
        
        self.network_weights = self._initialize_weights()
        self.beta = self.network_weights['weights_gener']['h2']

        self.forward(self.x)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.get_variable('h1', [self.config['vocab_size'], self.config['layer1']]),
            'h2': tf.get_variable('h2', [self.config['layer1'], self.config['layer2']]),
            'out': tf.get_variable('out', [self.config['layer2'], self.topic_num]),
        }
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([self.config['layer1']], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([self.config['layer2']], dtype=tf.float32)),
            'out': tf.Variable(tf.zeros([self.topic_num], dtype=tf.float32)),
        }
        all_weights['weights_gener'] = {
            'h2': tf.Variable(xavier_init(self.topic_num, self.config['vocab_size']))
        }
        all_weights['biases_gener'] = {
            'b2': tf.Variable(tf.zeros([self.config['vocab_size']], dtype=tf.float32))
        }
        return all_weights

    def encoder(self, x):
        weights = self.network_weights["weights_recog"]
        biases = self.network_weights['biases_recog']
        layer_1 = self.active_fct(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = self.active_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_do = tf.nn.dropout(layer_2, self.keep_prob)
        z_mean = tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out']), biases['out']))

        theta = tf.nn.softmax(z_mean)
        return theta

    def decoder(self, theta):
        x_recon = tf.contrib.layers.batch_norm(tf.add(tf.matmul(theta, self.network_weights["weights_gener"]['h2']), 0.0))
        x_recon = tf.nn.softmax(x_recon)
        return x_recon

    def negative_sampling(self, theta):
        logits = tf.cast(tf.less(theta, tf.reduce_min(tf.nn.top_k(theta, k=self.exclude_topt).values, axis=1, keepdims=True)), tf.float32)
        topic_indices = tf.one_hot(tf.multinomial(logits, self.select_topic_num), depth=theta.shape[1])  # N*1*K
        indices = tf.nn.top_k(tf.tensordot(topic_indices, self.beta, axes=1), self.word_sample_size).indices
        indices = tf.reshape(indices, shape=(-1, self.select_topic_num * self.word_sample_size))

        _m = tf.one_hot(indices, depth=self.beta.shape[1])
        _m = tf.reduce_sum(_m, axis=1)
        return _m

    def forward(self, x):
        self.theta_e = self.encoder(x)
        quantization_output = self.topic_dis_quant.forward(self.theta_e)
        self.theta_q = quantization_output['quantize']
        self.x_recon = self.decoder(self.theta_q)

        if self.word_sample_size > 0:
            print('==>word_sample_size > 0')
            _n_samples = self.negative_sampling(self.theta_q)
            negative_error = -self.w_omega * _n_samples * tf.log(1 - self.x_recon)
            self.auto_encoding_error = tf.reduce_mean(tf.reduce_sum(-x * tf.log(self.x_recon) + negative_error, axis=1))
            self.loss = self.auto_encoding_error + quantization_output["loss"]
        else:
            self.auto_encoding_error = tf.reduce_mean(tf.reduce_sum(-x * tf.log(self.x_recon), axis=1))
            self.loss = self.auto_encoding_error + quantization_output['loss']

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        self.train_op = optimizer.minimize(self.loss)

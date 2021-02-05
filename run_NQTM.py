# -*- coding:utf-8 -*-
import os
import numpy as np
import scipy.sparse
import tensorflow as tf
import os
import pickle
import argparse
from NQTM import NQTM

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--layer1', type=int, default=100)
parser.add_argument('--layer2', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--topic_num', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.002)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--word_sample_size', type=int, default=20)
parser.add_argument('--word_sample_epoch', type=int, default=150)
parser.add_argument('--omega', type=float, default=1.0)
parser.add_argument('--commitment_cost', type=float, default=0.1)
parser.add_argument('--test_index', type=int, default=1)
args = parser.parse_args()


def load_data(data_dir):
    train_data = scipy.sparse.load_npz(os.path.join(data_dir, 'bow_matrix.npz')).toarray()
    vocab = list()
    with open(os.path.join(data_dir, 'vocab.txt')) as file:
        for line in file:
            vocab.append(line.strip())
    return train_data, vocab


def create_minibatch(data):
    rng = np.random.RandomState(10)
    while True:
        ixs = rng.randint(data.shape[0], size=args.batch_size)
        yield data[ixs]


def print_top_words(beta, feature_names, n_top_words=15):
    top_words = list()
    for i in range(len(beta)):
        top_words.append(" ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
        print(top_words[-1])

    with open(os.path.join(args.output_dir, 'top_words_T{}_K{}_{}th'.format(n_top_words, args.topic_num, args.test_index)), 'w') as file:
        for line in top_words:
            file.write(line + '\n')


def get_theta(model, x):
    data_size = x.shape[0]
    batch_size = args.batch_size
    train_theta = np.zeros((data_size, args.topic_num))
    for i in range(int(data_size / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        data_batch = x[start:end]
        train_theta[start:end] = model.sess.run(model.theta_e, feed_dict={model.x: data_batch})
    train_theta[-batch_size:] = model.sess.run(model.theta_e, feed_dict={model.x: x[-batch_size:]})
    return train_theta


def train(model, train_data, vocab, config):
    total_batch = int(train_data.shape[0] / args.batch_size)
    minibatches = create_minibatch(train_data)
    op = [model.train_op, model.loss]

    for epoch in range(args.epoch):
        omega = 0 if epoch < config['word_sample_epoch'] else 1.0
        train_loss = list()
        for i in range(total_batch):
            batch_data = minibatches.__next__()
            feed_dict = {model.x: batch_data, model.w_omega: omega}
            _, batch_loss = model.sess.run(op, feed_dict=feed_dict)
            train_loss.append(batch_loss)

        print('Epoch: ', '{:03d} loss: {:.3f}'.format(epoch + 1, np.mean(train_loss)))

    beta = model.sess.run((model.beta))
    print_top_words(beta, vocab)

    train_theta = get_theta(model, train_data)
    np.save(os.path.join(args.output_dir, 'theta_K{}_{}th'.format(args.topic_num, args.test_index)), train_theta)
    np.save(os.path.join(args.output_dir, 'beta_K{}_{}th'.format(args.topic_num, args.test_index)), beta)


if __name__ == "__main__":
    config = dict()
    config.update(vars(args))
    config['active_fct'] = tf.nn.softplus

    os.makedirs(args.output_dir, exist_ok=True)

    train_data, vocab = load_data(args.data_dir)
    config['vocab_size'] = len(vocab)
    model = NQTM(config=config)
    train(model, train_data, vocab, config)

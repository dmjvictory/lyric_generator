#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
########################################################################

"""
File: gen_lyrics.py
Author: dengmingjie(dengmingjie@baidu.com)
Date: 2017/01/24 11:18:47
"""

import os
import sys
import time
import math
import jieba
import random
import codecs
import numpy as np
from collections import Counter
from collections import defaultdict
from ConfigParser import SafeConfigParser

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq as seq2seq
from tensorflow.contrib.tensorboard.plugins import projector
PAD_ID, GO_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

def get_config(config_file = 'seq2seq_model.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)

    for key, value in parser.items('ints'):
        tf.app.flags.DEFINE_integer(key, int(value), '')
    for key, value in parser.items('floats'):
        tf.app.flags.DEFINE_float(key, float(value), '')
    
    return tf.app.flags.FLAGS

FLAGS = get_config()

class DataGenerator():

    def __init__(self, filename):
        """
        param initialization
        """
        self.context = ''
        self.read_data(filename)
        self.gen_batches()

    def read_data(self, filename, _cut=False):
        for line in codecs.open(filename, 'r', 'utf8').readlines():
            self.context += line.strip('\n,') + u'。'
        self.vocab = Counter(self.context)
        self.vocab_list = map(lambda x:x[0], self.vocab.most_common())

        print 'vocab size:', len(self.vocab_list)
         
        self.char2id_dict = {w: i + 4 for i, w in enumerate(self.vocab_list)}
        self.id2char_dict = {i + 4: w for i, w in enumerate(self.vocab_list)}
        for i, item in enumerate(['pad', 'go', 'eos', 'unk']):
            self.char2id_dict[item] = i
            self.id2char_dict[i] = item

    def char2id(self, c):
        return self.char2id_dict.get(c, UNK_ID)

    def id2char(self, id):
        return self.id2char_dict.get(id, 'unk')

    def save_metadata(self, file):
        with open(file, 'w') as f:
            f.write('id\tchar\n')
            for i in range(self.vocab_size):
                c = self.id2char(i)
                f.write('{}\t{}\n'.format(i, c.encode('utf8')))

    def gen_batches(self):
        self.context = np.array([self.char2id(_) for _ in self.context])
        self.num_batches = int(self.context.size / (FLAGS.batch_size * FLAGS.seq_length))
        self.context = self.context[:self.num_batches * FLAGS.batch_size * FLAGS.seq_length]
        print 'batch number:', self.num_batches
        xdata = self.context
        ydata = np.copy(self.context)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        #np.split: data, indices, axis
        self.x_batches = np.split(xdata.reshape(FLAGS.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(FLAGS.batch_size, -1), self.num_batches, 1)
        self.pointer = 0 

    def next_batch(self):
        ''' 
        pointer for outputing mini-batches when training
        '''
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        if self.pointer == self.num_batches:
            self.pointer = 0
        return x, y
 

class Model(object):
    def __init__(self, infer=0):
        if infer:
            FLAGS.batch_size = 1
            FLAGS.seq_length = 1
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(
                tf.int32, [FLAGS.batch_size, FLAGS.seq_length])
            self.target_data = tf.placeholder(
                tf.int32, [FLAGS.batch_size, FLAGS.seq_length])
        
        with tf.name_scope('model'):
            self.cell = rnn.MultiRNNCell(
                       [rnn.BasicLSTMCell(FLAGS.cell_size) for _ in range(FLAGS.num_layers)])
            #self.cell = rnn.DropoutWrapper(self.cell, output_keep_prob=FLAGS.keep_prob)
            
            self.initial_state = self.cell.zero_state(
                FLAGS.batch_size, tf.float32)
            self.attention_state = tf.truncated_normal(
                [FLAGS.batch_size, FLAGS.attn_length, FLAGS.attn_size],
                 stddev=0.1, dtype=tf.float32)

            with tf.variable_scope('rnnlm'):
                # output weights
                softmax_w = tf.get_variable(
                    'softmax_w', [FLAGS.cell_size, FLAGS.vocab_size])
                softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])
                # embedding weights
                embedding = tf.get_variable(
                    'embedding', [FLAGS.vocab_size, FLAGS.embedding_size])
                inputs = tf.nn.embedding_lookup(embedding, self.input_data)
                inputs = tf.split(inputs, FLAGS.seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs] #(batch_size * embedding_size) 
                def loop(prev, _):
                    prev = tf.matmul(prev, softmax_w) + softmax_b
                    # select the best output and stop the gradient
                    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1)) 
                    return tf.nn.embedding_lookup(embedding, prev_symbol)
         
                outputs, last_state = seq2seq.attention_decoder(
                    inputs, self.initial_state, self.attention_state, self.cell, 
                    loop_function=loop if infer else None, scope='rnnlm') #(batch_size * rnn_size)

        with tf.name_scope('loss'):
            output = tf.reshape(tf.concat(outputs, 1), [-1, FLAGS.cell_size]) 
            #([(batch_size*seq_length) * rnn_size])

            self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            targets = tf.reshape(self.target_data, [-1])
            loss = seq2seq.sequence_loss_by_example([self.logits],
                                                    [targets],
                                                    [tf.ones_like(targets, dtype=tf.float32)])
            self.cost = tf.reduce_sum(loss) / FLAGS.batch_size / FLAGS.seq_length
            tf.summary.scalar('loss', self.cost) 

        with tf.name_scope('optimize'):
            self.lr = tf.Variable(0.0, trainable=False)
            tf.summary.scalar('learning_rate', self.lr)

            optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            for g in grads:
                tf.summary.histogram(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, FLAGS.max_gradient_norm)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.merged_op = tf.summary.merge_all()
    
        print tf.trainable_variables()

def train(data, model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        previous_losses = []
        learning_rate = FLAGS.learning_rate
        max_iter = FLAGS.n_epoch * data.num_batches

        for i in range(max_iter):
            x_batch, y_batch = data.next_batch()
            feed_dict = {model.input_data: x_batch,
                         model.target_data: y_batch,
                         model.lr: learning_rate}
            
            output_list = [model.cost, model.train_op]
            train_loss, _ = sess.run(output_list, feed_dict)

            if i % 50 == 0:
                print 'Epoch: {}/{}, Step:{}/{}, training_loss: {:4f}, learning_rate: {:4f}'.format(
                        i / data.num_batches, FLAGS.n_epoch, 
                        i, max_iter, train_loss, learning_rate)
                #if len(previous_losses) > 2 and train_loss > max(previous_losses[-3:]):
                #if i % 100 == 0:
                    #learning_rate = learning_rate * FLAGS.learning_rate_decay_factor

            if i % 2000 == 0 or (i + 1) == max_iter:
                saver.save(sess, os.path.join(
                    'model/', 'lyrics_model.ckpt'), global_step=i)

            previous_losses.append(train_loss)
            sys.stdout.flush() 


def sample(data, model):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint('model/')
        saver.restore(sess, ckpt)

        # initial phrase to warm RNN
        prime = u'我们'
        prime = [w for w in prime]
        state = sess.run(model.cell.zero_state(1, tf.float32))
        attention_state = sess.run(
            tf.truncated_normal([1, FLAGS.attn_length, FLAGS.attn_size],
                                stddev=0.1, dtype=tf.float32))

        for word in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = data.char2id(word)
            feed_dict = {model.input_data: x, 
                         model.initial_state: state, 
                         model.attention_state: attention_state}
            probs, state, attention_state = sess.run([model.probs, 
                        model.last_state, model.attention_state], feed_dict)

        def pick(p, word, sampling_type):
            def weighted_pick(weights):
                t = np.cumsum(weights)
                s = np.sum(weights)
                return(int(np.searchsorted(t, np.random.rand(1)*s)))

            if sampling_type == 'argmax':
                sample = np.argmax(p)
            elif sampling_type == 'weighted': 
                sample = weighted_pick(p)
            elif sampling_type == 'combined':
                if word in u',。':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            return sample
        
        words = []
        word = prime[-1]
        for i in range(200):
            x = np.zeros([1, 1])
            x[0, 0] = data.char2id(word)
            feed_dict = {model.input_data: x, 
                         model.initial_state: state,
                         model.attention_state: attention_state}
            probs, state, attention_state = sess.run([model.probs, 
                        model.last_state, model.attention_state], feed_dict)
            p = probs[0]
            word = data.id2char(pick(p, word, 'combined'))
            
            words.append(word.encode('utf8'))
        print ''.join([_.encode('utf8') for _ in prime]) + ''.join(words)


if __name__ == '__main__':
    data = DataGenerator(sys.argv[1])
    model = Model(infer=1)
    #train(data, model)
    sample(data, model)


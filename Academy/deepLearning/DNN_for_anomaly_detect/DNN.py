from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pandas as pd

##
'''
DNN
최초작성 > 2017-12-12 김지환 
'''

class DNN:
    @staticmethod
    def DNN(x, input_nodes, pkeep):
      print('@@ DNN @@')

      # Multiplier maintains a fixed ratio of nodes between each layer.
      mulitplier = 1.5

      # Number of nodes in each hidden layer
      hidden_nodes1 = 18
      hidden_nodes2 = round(hidden_nodes1 * mulitplier)
      hidden_nodes3 = round(hidden_nodes2 * mulitplier)

      # layer 1
      W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev=0.15))
      b1 = tf.Variable(tf.zeros(hidden_nodes1))
      y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

      # layer 2
      W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev=0.15))
      b2 = tf.Variable(tf.zeros(hidden_nodes2))
      y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

      # layer 3
      W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev=0.15))
      b3 = tf.Variable(tf.zeros([hidden_nodes3]))
      y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)
      y3 = tf.nn.dropout(y3, pkeep)

      # layer 4
      W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, 2], stddev=0.15))
      b4 = tf.Variable(tf.zeros([2]))
      y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)

      # output
      y = y4

      return y
    
  
    def __init__(self,
                 input_dim,
                 output_dim,
                 name='dnn',
                 loss='cross',
                 opt='adam'):
        '''
        DNN 모델의 입력차수, 출력차수에 따라 모델구성 변수들을 생성하고,
        학습의 오차(loss) 계산식 정의, 최적화 함수 정의를 한다.

        :param input_dim:
        :param output_dim:
        :param name:
        :param loss:
        :param opt:
        '''
        print('@@ __init__ @@')

        # Network Parameters
        self.input_dim = input_dim #
        self.output_dim = output_dim #

        self.valid_x = None
        self.valid_y = None
        self.valid_stop = 0 # 학습종료 판단하는 validation 기준값

        self.learning_rate = 0.005  # 하이퍼파라미터
        self.fig_num = 0 # for plotting

        # clear all things in tensorflow
        tf.reset_default_graph()

        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.output_dim])
        # Percent of nodes to keep during dropout.
        self.pkeep = tf.placeholder(tf.float32)

        # self.prediction == DNN
        self.prediction = DNN.DNN(self.X, self.input_dim, self.pkeep)

        ## Define loss and optimizer
        print('LOSS:', loss, ' OPT:', opt)
        if loss == 'abs':
          self.loss_op = tf.reduce_mean(tf.abs(self.prediction - self.Y))
        elif loss == 'cross':
          self.loss_op = -tf.reduce_sum(self.Y * tf.log(self.prediction))
        else:
          self.loss_op = -tf.reduce_sum(self.Y * tf.log(self.prediction))

        if opt == 'grad':
          self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        elif opt == 'adam':
          self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        elif opt == 'rms':
          self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
        else:
          assert(False)
          
        # self.train_op = self.optimizer.minimize(self.loss_op)
        
        # Evaluate model (with test logits, for dropout to be disabled)
        self.name_network = self.build_model_name(name)
        self.training_stop = None
        # Correct prediction if the most likely value (Fraud or Normal) from softmax equals the target value. 정확도
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.predict_by_nn = tf.nn.softmax(self.prediction)
        self.decision_by_nn = tf.argmax(self.prediction, 1)
        self.actual = tf.argmax(self.Y, 1)

    def set_name(self, name):
        self.name_network = self.build_model_name(name)
  
    def build_model_name(self, name):
        return '%s-I%d-O%d' % \
               (name, self.input_dim,
			self.output_dim)
    
    def save(self):
        fname = '%s%s.ckpt' % (CFG.NNMODEL, self.name_network)
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.sess, fname)
        print("Model saved in file: %s" % save_path)
    
    def load(self):
        fname = '%s%s.ckpt' % (CFG.NNMODEL, self.name_network)
        if not os.path.isfile(fname+'.index'):
          print('Model NOT found', fname)
          return False
        
        # Run the initializer
        self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, fname)
        print("Model restored from file: %s" % fname)
        return True
    
    def set_training_stop(self, training_stop):
      self.training_stop = training_stop
      
    def set_validation_data(self, valid_x, valid_y, valid_stop=0):
        print('@@ set_validation_data @@')
        if (valid_x is None) or (valid_x.shape[0] == 0):
          return
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.valid_stop = valid_stop # validation 결과값이 valid_stop 값 이하이면 학습 종료
      
    def do_validation(self):
        if self.valid_x is None or self.valid_x.shape[0] == 0:
          return 0
        valid_acc = self.do_test(self.valid_x, self.valid_y, 'Validation')
        return valid_acc
        
    
    def run(self, inputX, inputY, inputX_valid, inputY_valid, epochs=5, batch_size=2048, display_step=1, n_samples=0):
        '''
        :param inputX:
        :param inputY:
        :param inputX_valid:
        :param inputY_valid:
        :param epochs:
        :param batch_size:
        :param display_step:
        :param n_samples:
        :return:
        '''
        print('@@ run @@')

        # Parameters
        training_epochs = epochs  # should be 2000, it will timeout when uploading
        training_dropout = 0.9  # drop out
        display_step = display_step  # 10
        n_samples = n_samples
        batch_size = batch_size

        accuracy_summary = []  # Record accuracy values for plot
        cost_summary = []  # Record cost values for plot
        valid_accuracy_summary = []
        valid_cost_summary = []
        stop_early = 0  # To keep track of the number of epochs before early stopping

        print('@@ Session @@')
        # Run the initializer
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            for batch in range(int(n_samples / batch_size)):
                # print('batch :', batch)
                batch_x = inputX[batch * batch_size: (1 + batch) * batch_size]
                batch_y = inputY[batch * batch_size: (1 + batch) * batch_size]

                self.sess.run([self.optimizer], feed_dict={self.X: batch_x, self.Y: batch_y, self.pkeep: training_dropout})

            # Display logs after every 10 epochs
            if (epoch) % display_step == 0:
                train_accuracy, newCost = self.sess.run([self.accuracy, self.loss_op],
                                                   feed_dict={self.X: inputX, self.Y: inputY, self.pkeep: training_dropout})
                valid_accuracy, valid_newCost = self.sess.run([self.accuracy, self.loss_op],
                                                         feed_dict={self.X: inputX_valid, self.Y: inputY_valid, self.pkeep: 1})
                print("Epoch:", epoch,
                      'Acc  = ', "{:.5f}".format(train_accuracy),
                      'Cost = ', '{:.5f}'.format(newCost),
                      'Valid_Acc =', '{:.5f}'.format(valid_accuracy),
                      'Valid_Cost =', '{:.5f}'.format(valid_newCost)
                      )

                # Record the results of the model
                accuracy_summary.append(train_accuracy)
                cost_summary.append(newCost)
                valid_accuracy_summary.append(valid_accuracy)
                valid_cost_summary.append(valid_newCost)

                # If the model does not improve after 15 logs, stop the training.
                if valid_accuracy < max(valid_accuracy_summary) and epoch > 100:
                    stop_early += 1
                    if stop_early == 15:
                        break
                else:
                    stop_early = 0

        print()
        print("Optimization Finished!")
        print()
        
        return accuracy_summary, cost_summary, valid_accuracy_summary, valid_cost_summary

    def do_test(self, test_x, test_y, mesg='Test'):
        print('@@ do_test @@')
        test_predict_by_nn, test_decision_by_nn, test_actual = self.sess.run(
            [self.predict_by_nn, self.decision_by_nn, self.actual],
            feed_dict={self.X: test_x, self.Y: test_y,
                       self.pkeep: 1})

        print('test_predict_by_nn  :', test_predict_by_nn)
        print('test_decision_by_nn :', test_decision_by_nn)
        print('test_actual         :', test_actual)

        res = [(a[0], a[1], b, c) for a, b, c in zip(test_predict_by_nn, test_decision_by_nn, test_actual)]
        resdf = pd.DataFrame(data=res, columns=['Fr', 'Nm', 'NN', 'ACTUAL'])
        print(resdf.columns)

        TP = resdf[(resdf.NN == 0) & (resdf.ACTUAL == 0)].values.shape[0]
        FP = resdf[(resdf.NN == 0) & (resdf.ACTUAL == 1)].values.shape[0]
        TN = resdf[(resdf.NN == 1) & (resdf.ACTUAL == 1)].values.shape[0]
        FN = resdf[(resdf.NN == 1) & (resdf.ACTUAL == 0)].values.shape[0]

        print('TP', TP)
        print('FP', FP)
        print('TN', TN)
        print('FN', FN)
        print('Acc', (TP + TN) / (TP + TN + FP + FN))
        print('Precision', TP / (TP + FP))
        print('Recall', TP / (TP + FN))
        resdf.to_csv('predict.csv', index=False)
        # print(resdf[:100])

        return test_predict_by_nn, test_decision_by_nn, test_actual
    
    def do_compare(self, test_x, test_y):
        predict_y = self.sess.run(self.prediction, feed_dict={self.X: test_x, self.Y: test_y})
        diff = np.abs(predict_y - test_y) / test_y
        return diff
      
    def close(self):
      self.sess.close()
      




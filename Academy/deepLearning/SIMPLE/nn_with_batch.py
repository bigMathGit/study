import tensorflow as tf
import numpy as np
import random

def myfunc(_x):
  w = 1.3 # 기울기
  b = 2.6 # y 절편.  점(0, 2.6)
  # x 절편은 점(-2, 0)이 됨.
  _y = w * _x + b
  noise = random.random() * 0.01
  return _y + noise

# random.random() -- 0.0 ~ 1.0
NUM_DATA = 100
XVALUE = 3 # x값의 범위
# type: python list
xlist = [random.random() * XVALUE for i in range(NUM_DATA)]
ylist = [myfunc(x) for x in xlist]
print(xlist)
print(ylist)

# type: numpy ndarray
xlist = np.array(xlist)
ylist = np.array(ylist)
print(xlist.shape)  # shape ==  (10,)
print(ylist.shape)  # shape ==  (10,)
xlist = xlist.reshape((NUM_DATA, 1))  # shape ==  (10,1)
ylist = ylist.reshape((NUM_DATA, 1))  # shape ==  (10,1)
print(xlist.shape)
print(ylist.shape)

X = tf.placeholder(tf.float32, [None, 1], name='inputPlace')
y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([1,1], -1, 1), name='weight')
b = tf.Variable(tf.random_normal([1], -1, 1), name='bias')
O = tf.matmul(X, W) + b
O_ = tf.nn.sigmoid(O)
calc_error = tf.reduce_mean(tf.square(O - y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
training = optimizer.minimize(calc_error)

print('X', X.name)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
BATCH_SIZE = int(NUM_DATA / 20) # 한번의 training에 넣는 데이터의 개수.
for i in range(1000):
  # random sampling from 0,1,2,...,N-1
  index_selected = random.sample(range(NUM_DATA), BATCH_SIZE)
  batch_x = [xlist[i] for i in index_selected]
  batch_y = [ylist[i] for i in index_selected]

  res_training, error_val  = sess.run([training, calc_error],
                                      feed_dict={X: batch_x, y:batch_y})
  #print('RES_OPT', res_opt)
 
  if error_val < 0.00001:
    break
  if i % 10 == 0:
    see_loss = sess.run([calc_error],
                        feed_dict={X: xlist, y: ylist})
    see_o, see_w, see_b = sess.run([O, W, b], feed_dict={X: xlist, y: ylist})
    print('[%03d]' % i, end=' ')
    print('LOSS', see_loss, end=' ')
    print('W', see_w, 'bias', see_b)

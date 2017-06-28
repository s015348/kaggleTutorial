import sys
import numpy as np
import time
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf8')
print(sys.version)
print(tf.__version__)
#print(config)

vlen = 100000
iters = 10000

with tf.device(""):
    x = tf.placeholder(tf.float32, vlen)
    for i in range(iters):
        y = tf.exp(x)

with tf.Session() as sess:
    t0 = time.time()
    result = sess.run(y, feed_dict={x: np.random.random(vlen)})
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print(result)
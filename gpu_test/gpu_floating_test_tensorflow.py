import numpy as np
import time
import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

iters = 100
vlen = 10 * 30 * 768
x = tf.placeholder(tf.float32, vlen)
# not necessary
#y1 = tf.placeholder(tf.float32, vlen/2)
#y2 = tf.placeholder(tf.float32, vlen/2)
#y = tf.placeholder(tf.float32, vlen/2)

with tf.device("/job:local/task:1"):
    first_batch = tf.slice(x, [0], [vlen/2])
    for i in range(iters):
        y2 = tf.exp(x)

with tf.device("/job:local/task:0"):
    second_batch = tf.slice(x, [vlen/2], [-1])
    for i in range(iters):
        y1 = tf.div(y2, x)
        print("Loop %d" % i)
    y = y1 + y2

with tf.Session("grpc://localhost:2222") as sess:
    t0 = time.time()
    result = sess.run(y, feed_dict={x: np.random.random(vlen)})
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print(result)
import sys
import numpy as np
import time
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf8')
print(sys.version)
print(tf.__version__)

vlen = 100000
iters = 10000

task_type = sys.argv[1]
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

if task_type == 'server':
    server_number = int(sys.argv[2])
    print("Starting server #{}".format(server_number))

    server = tf.train.Server(cluster, job_name="local", task_index=server_number)
    server.start()
    server.join()

else:
    device = sys.argv[2]
    print("Starting worker using %s" % device)
    
    x = tf.placeholder(tf.float32, vlen)
    # not necessary
    #y1 = tf.placeholder(tf.float32, vlen/2)
    #y2 = tf.placeholder(tf.float32, vlen/2)
    #y = tf.placeholder(tf.float32, vlen/2)

    if device == 'gpu':
        with tf.device("/job:local/task:1"):
            first_batch = tf.slice(x, [0], [vlen/2])
            for i in range(iters):
                y2 = tf.exp(x)
    
        with tf.device("/job:local/task:0"):
            second_batch = tf.slice(x, [vlen/2], [-1])
            for i in range(iters):
                y1 = tf.exp(x)
                # print("Loop %d" % i)
            y = y1 + y2

    elif device == 'cpu':
        with tf.device("/job:local/task:1"):
            with tf.device("/cpu:0"):
                first_batch = tf.slice(x, [0], [vlen/2])
                for i in range(iters):
                    y2 = tf.exp(x)
    
        with tf.device("/job:local/task:0"):
            with tf.device("/cpu:0"):   
                second_batch = tf.slice(x, [vlen/2], [-1])
                for i in range(iters):
                    y1 = tf.exp(x)
                    # print("Loop %d" % i)
                y = y1 + y2

    
    with tf.Session("grpc://localhost:2222") as sess:
        t0 = time.time()
        result = sess.run(y, feed_dict={x: np.random.random(vlen)})
        t1 = time.time()
        print("Looping %d times took %f seconds" % (iters, t1 - t0))
        print(result)
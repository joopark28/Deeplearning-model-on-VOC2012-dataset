import tensorflow as tf
import prepare
import sys
#sys.path.append('/home/jackey/private/project/ML/deeplearning/ImageNet/Prepare')

inputXFilePath = '/home/jackey/private/project/ML/data/VOC2012/Binary/input_xs.txt'
inputTagFilePath = '/home/jackey/private/project/ML/data/VOC2012/Binary/input_ys.txt'
xs = tf.placeholder(tf.float32,[None,500,500,3],name="x_in")
ys = tf.placeholder(tf.float32,[None,24],name="y_in")


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
def Weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0)
    return initial

def bias_variable(shape):
    initial = tf.constant(0, shape=shape)
    return tf.Variable(initial)

def fc_op(input_op,n_in,n_out,name):

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        return activation


def conv_op(input_op,kw,kh,n_in,n_out,dw,dh,name):

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
 #       p += [kernel, biases]
        return activation
def max_pool(input_op,kw,kh,dw,dh,padding='SAME',name='None'):
    return tf.nn.max_pool(input_op, ksize=[1, kw, kh, 1], strides=[1, dw, dh, 1],padding=padding)

def interface(images_xs):
    conv1 = conv_op(images_xs, 5, 5, 3, 32, 1, 1, name="conv1")
    pool1 = max_pool(conv1, 2, 2, 2, 2, name="pool1")

    conv2 = conv_op(pool1, 7, 7, 32, 64, 1, 1, name="conv2")
    pool2 = max_pool(conv2, 2, 2, 2, 2, name="pool2")

    conv3 = conv_op(pool2,9,9,64,128,1,1,name='conv3')
    pool3 = max_pool(conv3, 2, 2, 2, 2, name="pool3")

    conv4 = conv_op(pool3,3,3,128,256,1,1,name='conv4')
    pool4 = max_pool(conv4, 3, 3, 3, 3, name="pool4")

    conv5 = conv_op(pool4,5,5,256,512,1,1,name='conv5')
    pool5 = max_pool(conv5, 3, 3, 3, 3, name="pool5")

    conv6 = conv_op(pool5,6,6,512,1024,1,1,name='conv6')
    pool6 = max_pool(conv6, 3, 3, 3, 3, name="pool6")

    conv7 = conv_op(pool6,3,3,1024,2048,1,1,name='conv7')
    pool7 = max_pool(conv7, 3, 3, 1, 1, name="pool7",padding='VALID')

    flat_pool7 = tf.reshape(pool7, [-1,1*1*2048])
    print(flat_pool7)
    fc1 = fc_op(flat_pool7,2048,24,name='fc1')

    drop_fc1= tf.nn.dropout(fc1, 0.5)
    return drop_fc1
drop_fc1 = interface(images_xs=xs)
prediction = tf.nn.softmax(drop_fc1)
init =tf.global_variables_initializer()
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.Session() as sess:
    batch_xs,batch_ys = prepare.load_binary_file(inputXFilePath,inputTagFilePath)
    print(batch_xs.shape)
    print(prediction.shape)
    print(batch_ys.shape)
    init = tf.global_variables_initializer()
    sess.run(init)
    """
    for step in range(500):
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if step%50 ==0:
            print(compute_accuracy(batch_xs[:100],batch_ys[:100]))
"""
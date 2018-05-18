import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorboard as tb




CIFAR_DIR = 'cifar-10-batches-py/'

def unpickle(file):
    with open(file,'rb')  as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')

    return cifar_dict

dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

all_data = [0, 1, 2, 3, 4, 5, 6]

for i, direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta  = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

print(data_batch1.keys())

# Display single image using matplotlib

X = data_batch1[b"data"]

X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

# plt.imshow(X[0])
# plt.show()

def one_hot_encode(vec, vals = 2):   # 10 for ten labels
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class Cifar():
    def __init__(self):
        self.i = 0
        self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        self.test_batch = [test_batch]

        self.training_images =  None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        print("setting up Training images and labels")

        #vertically stacking the images
        self.training_images = np.vstack([d[b'data'] for d in self.all_train_batches])
        train_len = len(self.training_images)

        #reshaping and normalizing
        # divide by max value to normalize
        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0,2,3,1)/255
        self.training_labels = one_hot_encode(np.hstack(list(map(lambda x: 0 if x > 1 else 1, d[b"labels"])) for d in self.all_train_batches))



        print("setting up Test images and labels")

        self.test_images = np.vstack([d[b'data'] for d in self.test_batch])
        test_len = len(self.test_images)

        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0,2,3,1)/255
        self.test_labels = one_hot_encode(np.hstack(list(map(lambda x: 0 if x > 1 else 1, d[b"labels"])) for d in self.test_batch))


    def next_batch(self, batch_size):
        # from self.i upto self.i +  batch size

        x = self.training_images[self.i:self.i+batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i+batch_size) % len(self.training_images)
        return x,y


ch = Cifar()
ch.set_up_images()
# batch = ch.next_batch(100)

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 2])

hold_prob = tf.placeholder(tf.float32)


def init_wight(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_wight(shape=shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W)+b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_wight([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


convo_1 = convolutional_layer(x, shape=[4, 4, 3, 32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[4, 4, 32, 64])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 8*8*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, 2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

matches = tf.equal(tf.argmax(y_pred, 1), (tf.argmax(y_true, 1)))
accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('cost', cross_entropy)
merged = tf.summary.merge_all()


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        batch = ch.next_batch(100)
        sess.run(train, feed_dict={x:batch[0], y_true:batch[1], hold_prob:0.5})
        train_writer = tf.summary.FileWriter(f'./Users/Attinder/train/cifar10', sess.graph)
        test_writer = tf.summary.FileWriter(f'./Users/Attinder/test/cifar10', sess.graph)
        train_writer.flush()
        test_writer.flush()

        if i%10 ==0:
             print("STEP: {}".format(i))
             summary, acc, cost = sess.run([merged, accuracy, cross_entropy], feed_dict={x:ch.test_images, y_true:ch.test_labels, hold_prob: 1.0})
             test_writer.add_summary(summary, i)

             print(f'Step {i}; Accuracy: {acc}')

             #print(sess.run(accuracy, feed_dict={x:ch.test_images, y_true:ch.test_labels, hold_prob:1.0}))
             print("\n")

        else:
            batch = ch.next_batch(100)
            sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
            summary = sess.run(merged, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
            train_writer.add_summary(summary, i)





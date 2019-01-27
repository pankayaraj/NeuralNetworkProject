from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import subprocess


input_size = 784
output_size = 10


h1_size = 1024
learning_rate = 0.001
batch_size = 700
no_iteraion = 1000
dropout = 0.8


img_h = img_w = 28
img_size_flat = img_h * img_w
n_classes = 10

def write_sprite_image(filename, images):
    """
        Create a sprite image consisting of sample images
        :param filename: name of the file to save on disk
        :param shape: tensor of flattened images
    """

    # Invert grayscale image
    images = 1 - images

    # Calculate number of plot
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    # Make the background of sprite image
    sprite_image = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            img_idx = i * n_plots + j
            if img_idx < images.shape[0]:
                img = images[img_idx]
                sprite_image[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = img

    plt.imsave(filename, sprite_image, cmap='gray')
    print('Sprite image saved in {}'.format(filename))

def write_metadata(filename, labels):
    """
            Create a metadata file image consisting of sample indices and labels
            :param filename: name of the file to save on disk
            :param shape: tensor of labels
    """
    with open(filename, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("{}\t{}\n".format(index, label))

    print('Metadata file saved in {}'.format(filename))



def Conv2D(x, W, b, strides = 1):
    x = tf.nn.conv2d(input=x, filter=W,strides= [1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    a = tf.nn.relu(x)

    return a

def maxpool2D(x , k):
    return tf.nn.max_pool(x, ksize=[1, k , k, 1], strides=[1, k, k, 1],padding='SAME')

with tf.variable_scope('Input'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name="input")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, output_size], name="output")
    X_  =  tf.reshape(X, shape=[-1, 28, 28, 1])
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'hw1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, output_size]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'hb1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([output_size]))
}


with tf.variable_scope("Convolution"):
    # Convolution Layer
    conv1 = Conv2D(X_, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2D(conv1, k=2)
    # Convolution Layer
    conv2 = Conv2D(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2D(conv2, k=2)

with tf.variable_scope("fully_connected_layer"):
    #hidden layer fully connected
    h1_ = tf.reshape(conv2, [-1, weights['hw1'].get_shape().as_list()[0]])
    h1 = tf.add(tf.matmul(h1_, weights['hw1']), biases['hb1'])
    h1 = tf.nn.relu(h1)

    #dropout
    h1 = tf.nn.dropout(h1, dropout)

    output = tf.add(tf.matmul(h1, weights['out']), biases['out'])
    prediction = tf.nn.softmax(output)



with tf.variable_scope('Train'):
    with tf.variable_scope("loss"):
        loss = tf.losses.softmax_cross_entropy(logits=output, onehot_labels=Y)
        tf.summary.scalar('loss', loss)

    with tf.variable_scope("optimizer"):
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train = opt.minimize(loss)

    with tf.variable_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(Y, axis = 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float64))
            tf.summary.scalar('accuracy', accuracy)


merged = tf.summary.merge_all()

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
logs_path = "D:\PROJECT\log/3"

writer = tf.summary.FileWriter(logs_path)


x_test = mnist.test.images
y_test = mnist.test.labels
tensor_shape = (x_test.shape[0] , 3136) # [test_set , h1]
embedding_var = tf.Variable(tf.zeros(tensor_shape),
                            name='h1_embedding')
# assign the tensor that we want to visualize to the embedding variable
embedding_assign = embedding_var.assign(tf.cast(h1_, tf.float32))
    # Create a config object to write the configuration parameters
config = projector.ProjectorConfig()
# Add embedding variable
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = 'metadata.tsv'
# Specify where you find the sprite. -> we will create this image later
embedding.sprite.image_path = 'sprite_images.png'
embedding.sprite.single_image_dim.extend([img_w, img_h])

    # Write a projector_config.pbtxt in the logs_path.
    # TensorBoard will read this file during startup.
projector.visualize_embeddings(writer, config)

# Reshape images from vector to matrix
x_test_images = np.reshape(np.array(x_test), (-1, img_w, img_h))
# Reshape labels from one-hot-encode to index
x_test_labels = np.argmax(y_test, axis=1)
write_sprite_image(os.path.join(logs_path, 'sprite_images.png'), x_test_images)
write_metadata(os.path.join(logs_path, 'metadata.tsv'), x_test_labels)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    for i in range(no_iteraion):

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={X:batch_x, Y:batch_y})
        l, a = sess.run([loss, accuracy], feed_dict={X:batch_x, Y:batch_y} )



        s = sess.run(merged, feed_dict={X:batch_x, Y:batch_y})
        writer.add_summary(s , i)


        print("Step " + str(i)+ " Loss = " + str(l) + " Accuracy = " + str(a))
    #EMBEDDING VISUALIZATION
    x_test_h1 = sess.run(embedding_assign, feed_dict={X: x_test})
# Save the tensor in model.ckpt file
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(logs_path, "model.ckpt"), i)

    test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels})
    print(test_accuracy)


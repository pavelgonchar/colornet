import tensorflow as tf
import numpy as np
import glob
import sys
from matplotlib import pyplot as plt
from batchnorm import ConvolutionalBatchNormalizer

filenames = sorted(glob.glob("../colornet/*/*.jpg"))
batch_size = 1
num_epochs = 1e+9

global_step = tf.Variable(0, name='global_step', trainable=False)
phase_train = tf.placeholder(tf.bool, name='phase_train')
uv = tf.placeholder(tf.uint8, name='uv')


def read_my_file_format(filename_queue, randomize=False):
    reader = tf.WholeFileReader()
    key, file = reader.read(filename_queue)
    uint8image = tf.image.decode_jpeg(file, channels=3)
    uint8image = tf.random_crop(uint8image, (224, 224, 3))
    if randomize:
        uint8image = tf.image.random_flip_left_right(uint8image)
        uint8image = tf.image.random_flip_up_down(uint8image, seed=None)
    float_image = tf.div(tf.cast(uint8image, tf.float32), 255)
    return float_image


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=False)
    example = read_my_file_format(filename_queue, randomize=False)
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.shuffle_batch(
        [example], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch


def batch_norm(x, depth, phase_train):
    with tf.variable_scope('batchnorm'):
        ewma = tf.train.ExponentialMovingAverage(decay=0.9999)
        bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)
        update_assignments = bn.get_assigner()
        x = bn.normalize(x, train=phase_train)
    return x


def conv2d(_X, w, sigmoid=False, bn=False):
    with tf.variable_scope('conv2d'):
        _X = tf.nn.conv2d(_X, w, [1, 1, 1, 1], 'SAME')
        if bn:
            _X = batch_norm(_X, w.get_shape()[3], phase_train)
        if sigmoid:
            return tf.sigmoid(_X)
        else:
            _X = tf.nn.relu(_X)
            return tf.maximum(0.01 * _X, _X)


def colornet(_tensors):
    """
    Network architecture http://tinyclouds.org/colorize/residual_encoder.png
    """
    with tf.variable_scope('colornet'):
        # Bx28x28x512 -> batch norm -> 1x1 conv = Bx28x28x256
        conv1 = tf.nn.relu(tf.nn.conv2d(batch_norm(_tensors[
                           "conv4_3"], 512, phase_train),
            _tensors["weights"]["wc1"], [1, 1, 1, 1], 'SAME'))
        # upscale to 56x56x256
        conv1 = tf.image.resize_bilinear(conv1, (56, 56))
        conv1 = tf.add(conv1, batch_norm(
            _tensors["conv3_3"], 256, phase_train))

        # Bx56x56x256-> 3x3 conv = Bx56x56x128
        conv2 = conv2d(conv1, _tensors["weights"][
                       'wc2'], sigmoid=False, bn=True)
        # upscale to 112x112x128
        conv2 = tf.image.resize_bilinear(conv2, (112, 112))
        conv2 = tf.add(conv2, batch_norm(
            _tensors["conv2_2"], 128, phase_train))

        # Bx112x112x128 -> 3x3 conv = Bx112x112x64
        conv3 = conv2d(conv2, _tensors["weights"][
                       'wc3'], sigmoid=False, bn=True)
        # upscale to Bx224x224x64
        conv3 = tf.image.resize_bilinear(conv3, (224, 224))
        conv3 = tf.add(conv3, batch_norm(_tensors["conv1_2"], 64, phase_train))

        # Bx224x224x64 -> 3x3 conv = Bx224x224x3
        conv4 = conv2d(conv3, _tensors["weights"][
                       'wc4'], sigmoid=False, bn=True)
        conv4 = tf.add(conv4, batch_norm(
            _tensors["grayscale"], 3, phase_train))

        # Bx224x224x3 -> 3x3 conv = Bx224x224x3
        conv5 = conv2d(conv4, _tensors["weights"][
                       'wc5'], sigmoid=False, bn=True)
        # Bx224x224x3 -> 3x3 conv = Bx224x224x2
        conv6 = conv2d(conv5, _tensors["weights"][
                       'wc6'], sigmoid=True, bn=True)

    return conv6


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)

    return temp


def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.mul(yuv, 255)
    yuv2rgb_filter = tf.constant(
        [[[[1., 1., 1.],
           [0., -0.34413999, 1.77199996],
            [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp, tf.mul(
        tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp


with open("vgg/tensorflow-vgg16/vgg16-20160129.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

with tf.variable_scope('colornet'):
    # Store layers weight
    weights = {
        # 1x1 conv, 512 inputs, 256 outputs
        'wc1': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01)),
        # 3x3 conv, 512 inputs, 128 outputs
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01)),
        # 3x3 conv, 256 inputs, 64 outputs
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01)),
        # 3x3 conv, 128 inputs, 3 outputs
        'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01)),
        # 3x3 conv, 6 inputs, 3 outputs
        'wc5': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01)),
        # 3x3 conv, 3 inputs, 2 outputs
        'wc6': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01)),
    }

colorimage = input_pipeline(filenames, batch_size, num_epochs=num_epochs)
colorimage_yuv = rgb2yuv(colorimage)

grayscale = tf.image.rgb_to_grayscale(colorimage)
grayscale_rgb = tf.image.grayscale_to_rgb(grayscale)
grayscale_yuv = rgb2yuv(grayscale_rgb)
grayscale = tf.concat(3, [grayscale, grayscale, grayscale])

tf.import_graph_def(graph_def, input_map={"images": grayscale})

graph = tf.get_default_graph()

with tf.variable_scope('vgg'):
    conv1_2 = graph.get_tensor_by_name("import/conv1_2/Relu:0")
    conv2_2 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
    conv3_3 = graph.get_tensor_by_name("import/conv3_3/Relu:0")
    conv4_3 = graph.get_tensor_by_name("import/conv4_3/Relu:0")

tensors = {
    "conv1_2": conv1_2,
    "conv2_2": conv2_2,
    "conv3_3": conv3_3,
    "conv4_3": conv4_3,
    "grayscale": grayscale,
    "weights": weights
}

# Construct model
pred = colornet(tensors)
pred_yuv = tf.concat(3, [tf.split(3, 3, grayscale_yuv)[0], pred])
pred_rgb = yuv2rgb(pred_yuv)

loss = tf.square(tf.sub(pred, tf.concat(
    3, [tf.split(3, 3, colorimage_yuv)[1], tf.split(3, 3, colorimage_yuv)[2]])))

if uv == 1:
    loss = tf.split(3, 2, loss)[0]
elif uv == 2:
    loss = tf.split(3, 2, loss)[1]
else:
    loss = (tf.split(3, 2, loss)[0] + tf.split(3, 2, loss)[1]) / 2

if phase_train:
    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    opt = optimizer.minimize(
        loss, global_step=global_step, gate_gradients=optimizer.GATE_NONE)

# Summaries
tf.histogram_summary("weights1", weights["wc1"])
tf.histogram_summary("weights2", weights["wc2"])
tf.histogram_summary("weights3", weights["wc3"])
tf.histogram_summary("weights4", weights["wc4"])
tf.histogram_summary("weights5", weights["wc5"])
tf.histogram_summary("weights6", weights["wc6"])
tf.histogram_summary("instant_loss", tf.reduce_mean(loss))
tf.image_summary("colorimage", colorimage, max_images=1)
tf.image_summary("pred_rgb", pred_rgb, max_images=1)
tf.image_summary("grayscale", grayscale_rgb, max_images=1)

# Saver.
saver = tf.train.Saver()

# Create the graph, etc.
init_op = tf.initialize_all_variables()

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables.
sess.run(init_op)

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("tb_log", sess.graph_def)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        # Run training steps
        training_opt = sess.run(opt, feed_dict={phase_train: True, uv: 1})
        training_opt = sess.run(opt, feed_dict={phase_train: True, uv: 2})

        step = sess.run(global_step)

        if step % 1 == 0:
            pred_, pred_rgb_, colorimage_, grayscale_rgb_, cost, merged_ = sess.run(
                [pred, pred_rgb, colorimage, grayscale_rgb, loss, merged], feed_dict={phase_train: False, uv: 3})
            print {
                "step": step,
                "cost": np.mean(cost)
            }
            if step % 1000 == 0:
                summary_image = concat_images(grayscale_rgb_[0], pred_rgb_[0])
                summary_image = concat_images(summary_image, colorimage_[0])
                plt.imsave("summary/" + str(step) + "_0", summary_image)

            sys.stdout.flush()
            writer.add_summary(merged_, step)
            writer.flush()
        if step % 100000 == 99998:
            save_path = saver.save(sess, "model.ckpt")
            print("Model saved in file: %s" % save_path)
            sys.stdout.flush()

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()

import os, sys, time
sys.path.append(os.getcwd())
# make other directories visible to us
sys.path.append('../')

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import numpy as np
import tensorflow as tf
import argparse

import tflib
#import tflib.mnist
import tflib.plot
#import tflib.save_images
#import tflib.ops.batchnorm
#import tflib.ops.conv2d
#import tflib.ops.deconv2d
#import tflib.ops.linear

#from niftynet.layer.base_layer import TrainableLayer
#from niftynet.layer import layer_util
#from niftynet.layer.activation import ActiLayer, leaky_relu
#from niftynet.layer.bn import BNLayer
#from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
#from niftynet.layer.fully_connected import FCLayer, FullyConnectedLayer
#from niftynet.layer.deconvolution import DeconvolutionalLayer
#from niftynet.layer.elementwise import ElementwiseLayer

from util.data_loader import *
from util.parse_config import parse_config

class MnistWganInv(object):
    def __init__(self, x_dim=784, z_dim=64, latent_dim=64, batch_size=80,
                 c_gp_x=10., lamda=0.1, output_path='./'):
        self.x_dim = [-1] + x_dim[1:]
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.c_gp_x = c_gp_x
        self.lamda = lamda
        self.output_path = output_path

        self.gen_params = self.dis_params = self.inv_params = None

        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.x_p = self.generate(self.z)

        self.x = tf.placeholder(tf.float32, shape=x_dim)
        self.z_p = self.invert(self.x)
        #print_3 = tf.print("z_p: ", self.z_p)
        #with tf.control_dependencies([print_3]):
        #    self.z_p = tf.identity(self.z_p)
        #print_op = tf.print("z_p: ", self.z_p)
        #with tf.control_dependencies([print_op]):
        #    self.z_p = tf.identity(self.z_p)
        self.dis_x = self.discriminate(self.x)
        self.dis_x_p = self.discriminate(self.x_p)
        self.rec_x = self.generate(self.z_p)

        #print_op3 = tf.print('rec_x: ', self.rec_x)
        #with tf.control_dependencies([print_op3]):
        #    self.rec_x = tf.identity(self.rec_x)

        self.rec_z = self.invert(self.x_p)

        self.gen_cost = -tf.reduce_mean(self.dis_x_p)
        tf.summary.scalar('gen_cost', self.gen_cost)

        self.inv_cost = tf.reduce_mean(tf.square(self.x - self.rec_x))
        self.inv_cost += self.lamda * tf.reduce_mean(tf.square(self.z - self.rec_z))
        tf.summary.scalar('inv_cost', self.inv_cost)

        self.dis_cost = tf.reduce_mean(self.dis_x_p) - tf.reduce_mean(self.dis_x)
        tf.summary.scalar('dis_cost', self.dis_cost)

        alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
        difference = self.x_p - self.x
        interpolate = self.x + alpha * difference
        gradient = tf.gradients(self.discriminate(interpolate), [interpolate])[0]
        slope = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=1))
        gradient_penalty = tf.reduce_mean((slope - 1.) ** 2)
        self.dis_cost += self.c_gp_x * gradient_penalty

        self.gen_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Generator')
        self.inv_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Inverter')
        self.dis_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Discriminator')

        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
            self.gen_cost, var_list=self.gen_params)
        self.inv_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
            self.inv_cost, var_list=self.inv_params)
        self.dis_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
            self.dis_cost, var_list=self.dis_params)

    def generate(self, z):
        assert z.shape[1] == self.z_dim

        output = tf.layers.dense(
            z,
            self.latent_dim * 4 * 1 * 6 * 6,
            name='Generator.Input',
            reuse=tf.AUTO_REUSE)
        output = tf.nn.relu(output)
        print(output.shape)
        output = tf.reshape(output, [-1, 1, 6, 6, self.latent_dim * 4])  # [1, 3, 3, 4*64]
        print(output.shape)

        output = tf.layers.conv3d_transpose(
                    output,
                    filters=self.latent_dim*4,
                    kernel_size=8,
                    strides=(4, 4, 4),
                    padding='Same',
                    name='Generator.2',
                    reuse=tf.AUTO_REUSE)
        with tf.variable_scope('Generator.2', reuse=True):
            w = tf.get_variable('kernel')
            tf.summary.histogram('Generator.2/weights', w)

        #output = tf.keras.layers.Conv3DTranspose(filters=self.latent_dim * 2,
        #                                kernel_size=8,
        #                                strides=(4,4,4),
        #                                padding='SAME',
        #                                name='Generator.2')(output)
        output = tf.nn.relu(output)  # 4, 24, 24, 2*64
        print(output.shape)
        output = output[:, :3, :, :, :]
        print(output.shape)

        output = tf.layers.conv3d_transpose(
                    output,
                    filters=self.latent_dim*2,
                    kernel_size=4,
                    strides=(2, 2, 2),
                    padding='Same',
                    name='Generator.3',
                    reuse=tf.AUTO_REUSE)

        with tf.variable_scope('Generator.3', reuse=True):
            w = tf.get_variable('kernel')
            tf.summary.histogram('Generator.3/weights', w)
        output = tf.nn.relu(output)  #  6, 48, 48, 64
        print(output.shape)
        output = output[:, :5, :, :, :]
        print(output.shape)

        output = tf.layers.conv3d_transpose(
                    output,
                    filters=self.latent_dim,
                    kernel_size=8,
                    strides=(4, 4, 4),
                    padding='Same',
                    name='Generator.Out',
                    reuse=tf.AUTO_REUSE)
        print(output.shape)
        output = output[:, :19, :144, :144, :]
        print(output.shape)
        output = tf.nn.tanh(output)  # 28 x 28

        print(self.x_dim)
        return tf.reshape(output, self.x_dim)

    def discriminate(self, x):
        output = tf.reshape(x, self.x_dim)  # 28 x 28

        output = tf.layers.conv3d(
                    output,
                    filters=self.latent_dim,
                    kernel_size=(4,4,4),
                    strides=(2, 2, 2),
                    name='Discriminator.Input',
                    padding='Same',
                    reuse=tf.AUTO_REUSE)
        output = tf.nn.leaky_relu(output)  # 14 x 14

        output = tf.layers.conv3d(
                    output,
                    filters=self.latent_dim*2,
                    kernel_size=(4,4,4),
                    strides=(2, 2, 2),
                    name='Discriminator.1',
                    padding='Same',
                    reuse=tf.AUTO_REUSE)
        output = tf.nn.leaky_relu(output)  # 7 x 7

        output =  tf.layers.conv3d(
                    output,
                    filters=self.latent_dim*4,
                    kernel_size=(4,4,4),
                    strides=(2, 2, 2),
                    name='Discriminator.2',
                    padding='Same',
                    reuse=tf.AUTO_REUSE)
        output = tf.nn.leaky_relu(output)  # 4 x 4
        output = tf.reshape(output, [-1, self.latent_dim * 36])

        output = tf.layers.dense(
            output,
            1,
            name='Discriminator.Output',
            reuse=tf.AUTO_REUSE)
        output = tf.reshape(output, [-1])

        output = tf.nn.sigmoid(output)  # 7 x 7

        return output

    def invert(self, x):
        output = tf.reshape(x, self.x_dim)  # 28 x 28

        output =  tf.layers.conv3d(
                    output,
                    filters=self.latent_dim,
                    kernel_size=(8,8,8),
                    strides=(4, 4, 4),
                    name='Inverter.Input',
                    padding='Same',
                    reuse=tf.AUTO_REUSE)
        output = tf.nn.leaky_relu(output)  # 14 x 14

        output =  tf.layers.conv3d(
                    output,
                    filters=self.latent_dim*2,
                    kernel_size=(8,8,8),
                    strides=(4, 4, 4),
                    name='Inverter.1',
                    padding='Same',
                    reuse=tf.AUTO_REUSE)
        output = tf.nn.leaky_relu(output)  # 7 x 7

        output = tf.layers.conv3d(
                    output,
                    filters=self.latent_dim*4,
                    kernel_size=(8,8,8),
                    strides=(4, 4, 4),
                    name='Inverter.2',
                    padding='Same',
                    reuse=tf.AUTO_REUSE)
        output = tf.nn.leaky_relu(output)  # 4 x 4
        output = tf.reshape(output, [-1, self.latent_dim * 36])

        output = tf.layers.dense(
            output,
            self.latent_dim * 4,
            name='Inverter.3',
            reuse=tf.AUTO_REUSE)
        output = tf.nn.leaky_relu(output)

        output = tf.layers.dense(
            output,
            self.z_dim,
            name='Inverter.Output',
            reuse=tf.AUTO_REUSE)
        output = tf.reshape(output, [-1, self.z_dim])

        return output

    def train_gen(self, sess, x, z, merged):
        _gen_cost, _, summary = sess.run([self.gen_cost, self.gen_train_op, merged],
                                feed_dict={self.x: x, self.z: z})
        return _gen_cost, summary

    def train_dis(self, sess, x, z):
        _dis_cost, _ = sess.run([self.dis_cost, self.dis_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _dis_cost

    def train_inv(self, sess, x, z):
        _inv_cost, _ = sess.run([self.inv_cost, self.inv_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _inv_cost

    def generate_from_noise(self, sess, noise, frame):
        samples = sess.run(self.x_p, feed_dict={self.z: noise})
        for i in range(batch_size):
            #print("SAMPLES")
            #print(samples[i])
            save_array_as_nifty_volume(samples[i], "examples/samples_{0:}.nii.gz".format(frame + i))
        #tflib.save_images.save_images(
        #    samples.reshape((-1, 28, 28)),
        #    os.path.join(self.output_path, 'examples/samples_{}.png'.format(frame)))
        return samples

    def reconstruct_images(self, sess, images, frame):
        reconstructions = sess.run(self.rec_x, feed_dict={self.x: images})

        for i in range(batch_size):
            #print("RECONSTRUCTIONS")
            #print(reconstructions[i])
            save_array_as_nifty_volume(reconstructions[i], "examples/rec_{0:}.nii.gz".format(frame + i))
        for i in range(batch_size):
            save_array_as_nifty_volume(images[i], "examples/rec_gt_{0:}.nii.gz".format(frame + i))
        #tflib.save_images.save_images(
        #    comparison.reshape((-1, 28, 28)),
        #    os.path.join(self.output_path, 'examples/recs_{}.png'.format(frame)))
        return reconstructions


if __name__ == '__main__':
    tf.reset_default_graph()
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=12, help='dimension of z')
    parser.add_argument('--latent_dim', type=int, default=12,
                        help='latent dimension')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='training steps')
    parser.add_argument('--dis_iter', type=int, default=100,
                        help='discriminator steps')
    parser.add_argument('--c_gp_x', type=float, default=10.,
                        help='coefficient for gradient penalty x')
    parser.add_argument('--lamda', type=float, default=.1,
                        help='coefficient for divergence of z')
    parser.add_argument('--output_path', type=str, default='./',
                        help='output path')
    parser.add_argument('-config')
    args = parser.parse_args()
    config = parse_config(args.config)
    config_data = config['data']

    print("Loading data...")
    # dataset iterator
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    batch_size = config_data['batch_size']
    full_data_shape = [batch_size] + config_data['data_shape']
    #train_gen, dev_gen, test_gen = tflib.mnist.load(args.batch_size, args.batch_size)

    def inf_train_gen():
        while True:
            train_pair = dataloader.get_subimage_batch()
            tempx = train_pair['images']
            tempw = train_pair['weights']
            tempy = train_pair['labels']
            yield tempx, tempw, tempy

    fixed_images, _, _ = next(inf_train_gen())

    #_, _, test_data = tflib.mnist.load_data()
    #fixed_images = test_data[0][:32]
    #del test_data

    tf.set_random_seed(327)
    np.random.seed(327)
    fixed_noise = np.random.randn(64, args.z_dim)
    print("Initializing GAN...")
    mnistWganInv = MnistWganInv(
        x_dim=full_data_shape, z_dim=args.z_dim, latent_dim=args.latent_dim,
        batch_size=batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
        output_path=args.output_path)

    saver = tf.train.Saver(max_to_keep=1000)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('summaries/train', session.graph)

        images = noise = gen_cost = dis_cost = inv_cost = None
        dis_cost_lst, inv_cost_lst = [], []
        print("Starting training...")
        for iteration in range(args.iterations):
            for i in range(args.dis_iter):
                noise = np.random.randn(batch_size, args.z_dim)
                images, images_w, images_y = next(inf_train_gen())

                dis_cost_lst += [mnistWganInv.train_dis(session, images, noise)]
                inv_cost_lst += [mnistWganInv.train_inv(session, images, noise)]

            #print(images)
            gen_cost, summary = mnistWganInv.train_gen(session, images, noise, merged)
            train_writer.add_summary(summary, iteration)
            dis_cost = np.mean(dis_cost_lst)
            inv_cost = np.mean(inv_cost_lst)

            tflib.plot.plot('train gen cost', gen_cost)
            tflib.plot.plot('train dis cost', dis_cost)
            tflib.plot.plot('train inv cost', inv_cost)

            if iteration % 100 == 99:
                mnistWganInv.generate_from_noise(session, fixed_noise, iteration)
                mnistWganInv.reconstruct_images(session, fixed_images, iteration)

            if iteration % 1000 == 999:
                save_path = saver.save(session, os.path.join(
                    args.output_path, 'models/model'), global_step=iteration)

            #if iteration % 1000 == 999:
            #    dev_dis_cost_lst, dev_inv_cost_lst = [], []
            #    for dev_images, _ in dev_gen():
            #        noise = np.random.randn(batch_size, args.z_dim)
            #        dev_dis_cost, dev_inv_cost = session.run(
            #            [mnistWganInv.dis_cost, mnistWganInv.inv_cost],
            #            feed_dict={mnistWganInv.x: dev_images,
            #                       mnistWganInv.z: noise})
            #        dev_dis_cost_lst += [dev_dis_cost]
            #        dev_inv_cost_lst += [dev_inv_cost]
            #    tflib.plot.plot('dev dis cost', np.mean(dev_dis_cost_lst))
            #    tflib.plot.plot('dev inv cost', np.mean(dev_inv_cost_lst))

            if iteration < 5 or iteration % 100 == 99:
                tflib.plot.flush(os.path.join(args.output_path, 'models'))

            tflib.plot.tick()

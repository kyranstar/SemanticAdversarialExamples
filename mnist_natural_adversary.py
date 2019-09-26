import os, sys
import pickle
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from mri_wgan_inv_new import make_generator_model, make_inverter_model, Z_DIM

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')


import tflib.mnist
from mri_wgan_inv_new import make_generator_model
from search import iterative_search, recursive_search

def save_adversary(adversary, filename):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    ax[0].imshow(np.reshape(adversary['x'], (28, 28)),
                 interpolation='none', cmap=plt.get_cmap('gray'))
    ax[0].text(1, 5, str(adversary['y']), color='white', fontsize=50)
    ax[0].axis('off')

    ax[1].imshow(np.reshape(adversary['x_adv'], (28, 28)),
                 interpolation='none', cmap=plt.get_cmap('gray'))
    ax[1].text(1, 5, str(adversary['y_adv']), color='white', fontsize=50)
    ax[1].axis('off')

    fig.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_path', type=str, default='./models/model-47999',
                        help='mnist GAN path')
    parser.add_argument('--rf_path', type=str, default='./models/mnist_rf_9045.sav',
                        help='RF classifier path')
    parser.add_argument('--lenet_path', type=str, default='./models/cnn_mnist.h5',
                        help='LeNet classifier path')
    parser.add_argument('--classifier', type=str, default='lenet',
                        help='classifier: rf OR lenet')
    parser.add_argument('--search_type', type=str, default='iterative',
                        help='iterative, recursive, fgm, or PGD')
    parser.add_argument('--nsamples', type=int, default=5000,
                        help='number of samples in each search iteration')
    parser.add_argument('--niter', type=int, default=10,
                        help='number of iterations in gradient descent')
    parser.add_argument('--step', type=float, default=0.01,
                        help='Delta r for search step size, or eps for gradient descent')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_path', type=str, default='./examples/',
                        help='output path')
    args = parser.parse_args()


    if args.classifier == 'rf':
        classifier = pickle.load(open(args.rf_path, 'rb'), encoding='latin1')

        def cla_fn(x):
            return classifier.predict(np.reshape(x, (-1, 784)))

    elif args.classifier == 'lenet':
        graph_CLA = tf.Graph()
        sess_CLA = tf.compat.v1.Session(graph=graph_CLA)
        with graph_CLA.as_default():
            set_session(sess_CLA)
            classifier = load_model(args.lenet_path)

        def cla_fn(x):
            with graph_CLA.as_default():
                set_session(sess_CLA)
                return np.argmax(classifier.predict_on_batch(np.reshape(x, (-1, 28, 28))), axis=1)
        def logits_fn(x):
            with graph_CLA.as_default():
                set_session(sess_CLA)
                return classifier.predict_on_batch(np.reshape(x, (-1, 28, 28)))

    else:
        sys.exit('Please choose MNIST classifier: rf OR lenet')

    graph_GAN = tf.Graph()
    with graph_GAN.as_default():
        sess_GAN = tf.compat.v1.Session(graph=graph_GAN)
        set_session(sess_GAN)
        #sess_GAN.run(tf.compat.v1.global_variables_initializer())
        #model_generator =  load_model('models/generator.h5')
        #model_inverter =  load_model('models/inverter.h5')
        model_generator = make_generator_model()
        model_generator.build(input_shape=(Z_DIM,))
        model_generator.load_weights('models/generator.h5')
        model_inverter = make_inverter_model()
        model_inverter.build(input_shape=(1, 28,28, 1))
        model_inverter.load_weights('models/inverter.h5')
        #saver_GAN = tf.compat.v1.train.Saver(max_to_keep=100)
        #saver_GAN = tf.compat.v1.train.import_meta_graph('{}.meta'.format(args.gan_path))
        #saver_GAN.restore(sess_GAN, args.gan_path)
        #print("Restoring from", tf.train.latest_checkpoint(checkpoint_dir))
        #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        #checkpoint_dir = './training_checkpoints_mnist'
        #latest = tf.train.latest_checkpoint(checkpoint_dir)
        #model_GAN.load_weights(latest)


    def gen_fn(z):
        with sess_GAN.as_default():
            with graph_GAN.as_default():
                x_p = model_generator.predict(tf.cast(tf.constant(np.asarray(z)), 'float32'), steps=1)
        return x_p


    def inv_fn(x):
        with sess_GAN.as_default():
            with graph_GAN.as_default():
                z_p = model_inverter.predict(x)
        return z_p

    _, _, test_data = tflib.mnist.load_data()

    for i in range(10):
        y_t = None
        print("Creating adversary %d" % i)
        x = np.reshape(test_data[0][i], (1, 28, 28, 1))
        y = test_data[1][i]
        y_pred = cla_fn(x)[0]
        if y_pred != y:
            continue

        if args.search_type.lower() == 'iterative':
            adversary = iterative_search(gen_fn, inv_fn, cla_fn, x, y, y_t=y_t,
                               nsamples=args.nsamples, step=args.step, verbose=args.verbose)
        elif args.search_type.lower() == 'recursive':
            adversary = recursive_search(gen_fn, inv_fn, cla_fn, x, y, y_t=y_t,
                               nsamples=args.nsamples, step=args.step, verbose=args.verbose)
        #elif args.search_type.lower() == 'fgm':
        #    model = LatentToLabel(lambda z: model_GAN.generate(z), lambda x: classifier(x), graph_GAN, graph_CLA, sess_GAN, sess_GAN)
        #    adversary = fgm_search(model, gen_fn, inv_fn, cla_fn, sess_GAN, graph_GAN, x, y,
        #                       niter=args.niter, eps_iter=args.step, verbose=args.verbose)
        else:
            print("ERROR")
            exit()

        filename = 'mnist_{}_{}_{}.png'.format(str(i).zfill(4), args.search_type, args.classifier)

        save_adversary(adversary, os.path.join(args.output_path, filename))

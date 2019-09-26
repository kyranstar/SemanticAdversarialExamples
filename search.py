import numpy as np
#from attacks import FastGradientMethod, ProjectedGradientDescent
#from cleverhans.model import Model
import tensorflow as tf
from keras.layers import Convolution2D, Dense, Input
import keras

#class LatentToLabel(Model):
#    def __init__(self, gen_fn, classifier):
#        self.gen_fn = gen_fn
#        self.classifier = classifier

#    def fprop(self, z_pl, **kwargs):
#        """
#        Forward propagation to compute the model outputs.
#        :param z: A symbolic representation of the latent network input
#        :return: A dictionary mapping layer names to the symbolic
#                 representation of their output.
#        """
#        x = self.gen_fn(z_pl)
#        #self.classifier.summary()
#        # remove softmax layer
#        new_classifier = keras.models.Model(inputs=self.classifier.input, outputs=self.classifier.layers[-2].output)
#        #new_classifier.summary()
#        logits = new_classifier(tf.reshape(x, (-1, 1, 28, 28)))
        #new_classifier.summary()
        #newInput = Input(tensor=tf.reshape(x, (-1, 784)))
        #newOutputs = self.classifier(newInput)
        #self.classifier = Model(newInput, newOutputs)
#        return {Model.O_LOGITS: logits}

#def fgm_search(model, gen_fn, inv_fn, cla_fn, sess, x, y, y_t=None, z=None,
#                     niter=10, eps_iter=0.005, p=2, verbose=False):
#    """
#    Search using projected gradient descent
#    :param gen_fn: function of generator, G_theta
#    :param inv_fn: function of inverter, I_gamma
#    :param cla_fn: function of classifier, f
#    :param x: input instance
#    :param y: label
#    :param y_t: target label for adversary
#    :param z: latent vector corresponding to x
#    :param niter: number of iterations of pgd
#    :param eps_iter:
#    :param p: indicating norm order
#    :param verbose: print out
#    :return: adversary for x against cla_fn (d_adv is Delta z between z and z_adv)
#    """
#    y_og = y
#    y = tf.one_hot([y], 10)
#    if z is None:
#        z = inv_fn(x)

#    z_pl = tf.placeholder(tf.float32, shape=z.shape)
#    fgsm = ProjectedGradientDescent(model)
#    fgsm_params = {
#      'eps_iter': eps_iter,
#      'y': y,
#      'ord': p,
      #'y_t': y_t,
#      'loss_func': lambda logits, labels:tf.losses.softmax_cross_entropy(labels, logits)#print("%s, %s" % (str(labels), str(logits)))
      #'clip_min': 0.,
      #'clip_max': 1.
#    }
#    z_adv_op = fgsm.generate(z_pl, **fgsm_params)

#    z_adv = z
#    y_adv = y_og
#    counter = 0
#    while y_adv == y_og and counter < niter:
#        z_adv = sess.run(z_adv_op, {z_pl: z_adv})
#        x_adv = gen_fn(z_adv)
#        y_adv = cla_fn(x_adv)[0]
#        d_adv = np.linalg.norm(z_adv-z, ord=p)
#        counter += 1
#        print("Calculating z_adv... z distance = %f, classified as %d" % (d_adv, y_adv))
    #z_adv = z
    #preds_adv = model.get_logits(z_adv)
    #z_adv = z

#    adversary = {'x': x, 'y': y_og, 'z': z,
#                 'x_adv': x_adv, 'y_adv': y_adv, 'z_adv': z_adv, 'd_adv': d_adv}

#    return adversary

def iterative_search(gen_fn, inv_fn, cla_fn, x, y, y_t=None, z=None,
                     nsamples=5000, step=0.01, l=0., h=10., p=2, verbose=False):
    """
    Algorithm 1 in the paper, iterative stochastic search
    :param gen_fn: function of generator, G_theta
    :param inv_fn: function of inverter, I_gamma
    :param cla_fn: function of classifier, f
    :param x: input instance
    :param y: label
    :param y_t: target label for adversary
    :param z: latent vector corresponding to x
    :param nsamples: number of samples in each search iteration
    :param step: Delta r for search step size
    :param l: lower bound of search range
    :param h: upper bound of search range
    :param p: indicating norm order
    :param verbose: print out
    :return: adversary for x against cla_fn (d_adv is Delta z between z and z_adv)
    """
    x_adv, y_adv, z_adv, d_adv = None, None, None, None
    h = l + step

    def printout():
        if verbose and y_t is None:
            print("UNTARGET y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))
        elif verbose:
            print("TARGETED y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))

    if verbose:
        print("iterative search")

    if z is None:
        z = inv_fn(x)

    while True:
        delta_z = np.random.randn(nsamples, z.shape[1])     # http://mathworld.wolfram.com/HyperspherePointPicking.html
        d = np.random.rand(nsamples) * (h - l) + l          # length range [l, h)
        norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
        d_norm = np.divide(d, norm_p).reshape(-1, 1)        # rescale/normalize factor
        delta_z = np.multiply(delta_z, d_norm)
        z_tilde = z + delta_z       # z tilde
        x_tilde = gen_fn(z_tilde)   # x tilde
        y_tilde = cla_fn(x_tilde)   # y tilde

        if y_t is None:
            indices_adv = np.where(y_tilde != y)[0]
        else:
            indices_adv = np.where(y_tilde == y_t)[0]

        if len(indices_adv) == 0:       # no candidate generated
            l = h
            h = l + step
        else:                           # certain candidates generated
            idx_adv = indices_adv[np.argmin(d[indices_adv])]

            if y_t is None:
                assert (y_tilde[idx_adv] != y)
            else:
                assert (y_tilde[idx_adv] == y_t)

            if d_adv is None or d[idx_adv] < d_adv:
                x_adv = x_tilde[idx_adv]
                y_adv = y_tilde[idx_adv]
                z_adv = z_tilde[idx_adv]
                d_adv = d[idx_adv]
                printout()
                break

    adversary = {'x': x, 'y': y, 'z': z,
                 'x_adv': x_adv, 'y_adv': y_adv, 'z_adv': z_adv, 'd_adv': d_adv}

    return adversary


def recursive_search(gen_fn, inv_fn, cla_fn, x, y, y_t=None, z=None,
                     nsamples=5000, step=0.01, l=0., h=10., stop=5, p=2, verbose=False):
    """
    Algorithm 2 in the paper, hybrid shrinking search
    :param gen_fn: function of generator, G_theta
    :param inv_fn: function of inverter, I_gamma
    :param cla_fn: function of classifier, f
    :param x: input instance
    :param y: label
    :param y_t: target label for adversary
    :param z: latent vector corresponding to x
    :param nsamples: number of samples in each search iteration
    :param step: Delta r for search step size
    :param l: lower bound of search range
    :param h: upper bound of search range
    :param stop: budget of extra iterative steps
    :param p: indicating norm order
    :param verbose: print out
    :return: adversary for x against cla_fn (d_adv is Delta z between z and z_adv)
    """
    x_adv, y_adv, z_adv, d_adv = None, None, None, None
    counter = 1

    def printout():
        if verbose and y_t is None:
            print("UNTARGET y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f count=%d" % (y, y_adv, d_adv, l, h, counter))
        elif verbose:
            print("TARGETED y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f count=%d" % (y, y_adv, d_adv, l, h, counter))

    if verbose:
        print("first recursion")

    if z is None:
        z = inv_fn(x)

    while True:
        delta_z = np.random.randn(nsamples, z.shape[1])     # http://mathworld.wolfram.com/HyperspherePointPicking.html
        d = np.random.rand(nsamples) * (h - l) + l          # length range [l, h)
        norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
        d_norm = np.divide(d, norm_p).reshape(-1, 1)        # rescale/normalize factor
        delta_z = np.multiply(delta_z, d_norm)
        z_tilde = z + delta_z       # z tilde
        x_tilde = gen_fn(z_tilde)   # x tilde
        y_tilde = cla_fn(x_tilde)   # y tilde

        if y_t is None:
            indices_adv = np.where(y_tilde != y)[0]
        else:
            indices_adv = np.where(y_tilde == y_t)[0]

        if len(indices_adv) == 0:       # no candidate generated
            if h - l < step:
                break
            else:
                l = l + (h - l) * 0.5
                counter = 1
                printout()
        else:                           # certain candidates generated
            idx_adv = indices_adv[np.argmin(d[indices_adv])]

            if y_t is None:
                assert (y_tilde[idx_adv] != y)
            else:
                assert (y_tilde[idx_adv] == y_t)

            if d_adv is None or d[idx_adv] < d_adv:
                x_adv = x_tilde[idx_adv]
                y_adv = y_tilde[idx_adv]
                z_adv = z_tilde[idx_adv]
                d_adv = d[idx_adv]
                l, h = d_adv * 0.5, d_adv
                counter = 1
            else:
                h = l + (h - l) * 0.5
                counter += 1

            printout()
            if counter > stop or h - l < step:
                break

    if verbose:
        print('then iteration')

    if d_adv is not None:
        h = d_adv
    l = max(0., h - step)
    counter = 1
    printout()

    while counter <= stop and h > 1e-4:
        delta_z = np.random.randn(nsamples, z.shape[1])
        d = np.random.rand(nsamples) * (h - l) + l
        norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
        d_norm = np.divide(d, norm_p).reshape(-1, 1)
        delta_z = np.multiply(delta_z, d_norm)
        z_tilde = z + delta_z
        x_tilde = gen_fn(z_tilde)
        y_tilde = cla_fn(x_tilde)

        if y_t is None:
            indices_adv = np.where(y_tilde != y)[0]
        else:
            indices_adv = np.where(y_tilde == y_t)[0]

        if len(indices_adv) == 0:
            counter += 1
            printout()
        else:
            idx_adv = indices_adv[np.argmin(d[indices_adv])]

            if y_t is None:
                assert (y_tilde[idx_adv] != y)
            else:
                assert (y_tilde[idx_adv] == y_t)

            if d_adv is None or d[idx_adv] < d_adv:
                x_adv = x_tilde[idx_adv]
                y_adv = y_tilde[idx_adv]
                z_adv = z_tilde[idx_adv]
                d_adv = d[idx_adv]

            h = l
            l = max(0., h - step)
            counter = 1
            printout()

    adversary = {'x': x, 'y': y, 'z': z,
                 'x_adv': x_adv, 'y_adv': y_adv, 'z_adv': z_adv, 'd_adv': d_adv}

    return adversary


if __name__ == '__main__':
    pass

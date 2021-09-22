import numpy
import tensorflow as tf

import numpy as np

from optimizer.custom_lbfgs import lbfgs, Struct


def function_factory(model, loss_fcn, x, y, callback_fcn, epochs, x_test=None, y_test=None,
                     val_freq=1000, log_freq=1000, verbose=1):
    """
    A factory to create a function required by the L-BFGS implementation.

    :param tf.keras.Model model: an instance of `tf.keras.Model` or its subclasses
    :param object loss_fcn: a function with signature loss_value = loss(y_pred, y_true)
    :param tf.tensor x: input tensor of the training dataset
    :param tf.tensor y: output tensor of the training dataset
    :param object callback_fcn: callback function, which is called after each epoch
    :param int epochs: number of epochs
    :param tf.tensor x_test: input tensor of the test dataset, used to evaluate accuracy
    :param tf.tensor y_test: output tensor of the test dataset, used to evaluate accuracy
    :return: object: a function that has the signature of loss_value, gradients = f(model_parameters)
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = numpy.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(weights):
        """
        Updates the model's weights

        :param tf.Tensor weights: representing the model's weights
        """

        weights = tf.cast(weights, tf.float64)

        params = tf.dynamic_partition(weights, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    @tf.function
    def train_step(weights):
        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(weights)
            # calculate the loss
            loss_value = loss_fcn(y, model(x, training=True))

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        return loss_value, grads

    def f(weights):
        """
        Function that can be used in the L-BFGS implementation.
        This function is created by function_factory.

        :param tf.Tensor weights: representing the model's weights
        :return: tf.Tensor loss_value: current loss value, tf.Tensor grads: gradients w.r.t. the weights
        """
        loss_value, grads = train_step(weights)

        # print out iteration & loss
        f.iter += 1
        callback_fcn(f.iter, loss_value, epochs, x_test, y_test, val_freq=val_freq, log_freq=log_freq, verbose=verbose)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = 0
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


class LBFGS:
    """
    Class used to represent the L-BFGS optimizer.
    """

    def minimize(self, model, loss_fcn, x, y, callback_fcn, epochs=2000, learning_rate=1.,
                 x_test=None, y_test=None, val_freq=1000, log_freq=1000, verbose=1):
        """
        Performs the Neural Network training with the L-BFGS implementation.

        :param tf.keras.Model model: an instance of `tf.keras.Model` or its subclasses
        :param object loss_fcn: a function with signature loss_value = loss(y_pred, y_true)
        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param object callback_fcn: callback function, which is called after each epoch
        :param int epochs: number of epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """
        func = function_factory(model, loss_fcn, x, y, callback_fcn, epochs, x_test=x_test, y_test=y_test,
                                val_freq=val_freq, log_freq=log_freq, verbose=verbose)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)

        nt_epochs = epochs
        nt_config = Struct()
        nt_config.learningRate = learning_rate
        nt_config.maxIter = nt_epochs
        nt_config.nCorrection = 50
        nt_config.tolFun = 1.0 * np.finfo(float).eps

        lbfgs(func, init_params, nt_config, Struct(), True, lambda x, y, z: None)

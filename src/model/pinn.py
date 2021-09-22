import abc

import tensorflow as tf

from model.nn import NN


class PINN(NN, metaclass=abc.ABCMeta):
    """
    Class used to represent a Physics informed Neural Network, children of NN.
    """

    def __init__(self, layers, lb, ub):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bounds of the inputs of the training data
        :param np.ndarray ub: upper bounds of the inputs of the training data
        """

        super().__init__(layers, lb, ub)

        self.loss_object = self.loss

    def loss(self, y, y_pred):
        """
        Customized loss object to represent the composed mean squared error
        for Physics informed Neural Networks.
        Consists of the mean squared error between the predictions from the DNN (model) and reference values of the
        solution from the differential equation and the mean squared error of the predictions
        of the PINN (f_model) with zero.

        :param tf.tensor y: reference values of the solution of the differential equation
        :param tf.tensor y_pred: predictions of the solution of the differential equation
        :return: tf.tensor: composed mean squared error value
        """

        w_data = 1
        w_phys = 1

        f_pred = self.f_model()
        L_data = tf.reduce_mean(tf.square(y - y_pred))
        L_phys = tf.reduce_mean(tf.square(f_pred))

        L = w_data * L_data + w_phys * L_phys

        return L

    def f_model(self, x):
        """
        Declaration of the function for the implementation of the f_model for a specific differential equation.
        """
        pass

    def predict(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        return self.model.predict(x), self.f_model(x)

import abc
import logging
import os
import time
import datetime

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras import Sequential, backend
from tensorflow.python.keras.layers import Dense, InputLayer, Lambda

from optimizer.lbfgs import LBFGS
from utils.plotting import new_fig, save_fig

CHECKPOINTS_PATH = os.path.join('../checkpoints')


class NN(object, metaclass=abc.ABCMeta):
    """
    Abstract class used to represent a Neural Network.
    """

    def __init__(self, layers: list, lb: np.ndarray, ub: np.ndarray) -> None:
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bounds of the inputs of the training data
        :param np.ndarray ub: upper bounds of the inputs of the training data
        """

        self.checkpoints_dir = CHECKPOINTS_PATH

        self.dtype = "float64"
        # Descriptive Keras model
        backend.set_floatx(self.dtype)

        self.input_dim = layers[0]
        self.output_dim = layers[-1]

        # Keras Sequential Model
        self.model = Sequential()

        # Input Layer
        self.model.add(InputLayer(input_shape=(self.input_dim,)))

        # Normalization Layer
        self.model.add(Lambda(
            lambda X: 2.0 * (X - lb) / (ub - lb) - 1.0))

        # Hidden Layer
        for layer_width in layers[1:-1]:
            self.model.add(Dense(layer_width, activation=tf.nn.tanh,
                                 kernel_initializer='glorot_normal'))
        # Output Layer :
        self.model.add(Dense(self.output_dim))

        self.optimizer = None
        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.start_time = None
        self.prev_time = None

        # Store metrics
        self.train_loss_results = {}
        self.train_accuracy_results = {}
        self.train_time_results = {}
        self.train_pred_results = {}

    def tensor(self, X):
        """
        Converts a list or numpy array to a tf.tensor.

        :param list or nd.array X:
        :return: tf.tensor: tensor of X
        """
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def summary(self):
        """
        Pipes the Keras.model.summary function to the logging.
        """

        self.model.summary(print_fn=lambda x: logging.info(x))

    @tf.function
    def train_step(self, x, y):
        """
        Performs training step during training.

        :param tf.tensor x: (batched) input tensor of training data
        :param tf.tensor y: (batched) output tensor of training data
        :return: float loss: the corresponding current loss value
        """

        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = self.loss_object(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def fit(self, x, y, epochs=2000, x_test=None, y_test=None, optimizer='adam', learning_rate=0.1,
            load_best_weights=False, val_freq=1000, log_freq=1000, verbose=1):
        """
        Performs the neural network training phase.

        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate current accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate current accuracy
        :param str optimizer: name of the optimizer, choose from 'adam' or 'lbfgs'
        :param bool load_best_weights: flag to determine if the best weights corresponding to the best
        accuracy are loaded after training
        """

        x = self.tensor(x)
        y = self.tensor(y)

        self.start_time = time.time()
        self.prev_time = self.start_time

        if optimizer == 'adam':
            self.train_adam(x, y, epochs, x_test, y_test, learning_rate, val_freq, log_freq, verbose)
        elif optimizer == 'lbfgs':
            self.train_lbfgs(x, y, epochs, x_test, y_test, learning_rate, val_freq, log_freq, verbose)

        if load_best_weights is True:
            self.load_weights()

    def train_adam(self, x, y, epochs=2000, x_test=None, y_test=None, learning_rate=0.1, val_freq=1000, log_freq=1000,
                   verbose=1):
        """
        Performs the neural network training, using the adam optimizer.

        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')
        if verbose:
            logging.info(f'Start ADAM optimization')

        for epoch in range(1, epochs + 1):
            loss = self.train_step(x, y)
            # Track progress
            epoch_loss.update_state(loss)  # Add current batch loss

            self.epoch_callback(epoch, epoch_loss.result(), epochs, x_test, y_test, val_freq, log_freq,
                                verbose)

    def train_lbfgs(self, x, y, epochs=2000, x_test=None, y_test=None, learning_rate=1.0, val_freq=1000, log_freq=1000,
                    verbose=1):
        """
        Performs the neural network training, using the L-BFGS optimizer.

        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """

        # train the model with L-BFGS solver
        if verbose:
            logging.info(f'Start L-BFGS optimization')

        optimizer = LBFGS()
        optimizer.minimize(
            self.model, self.loss_object, x, y, self.epoch_callback, epochs, x_test=x_test, y_test=y_test,
            val_freq=val_freq, log_freq=log_freq, verbose=verbose, learning_rate=learning_rate)

    def predict(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        return self.model.predict(x)

    def plot_train_results(self, basename=None):
        """
        Visualizes the training metrics Loss resp. Accuracy over epochs.

        :param str basename: used to save the figure with this name, if None the figure is not saved
        """

        fig = new_fig()
        ax = fig.add_subplot(111)
        fig.suptitle(f'{self.name} - Training Metrics')

        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.plot(self.train_loss_results.keys(), self.train_loss_results.values(), label='Loss')
        if self.train_accuracy_results:
            ax.set_ylabel("Loss / Accuracy")
            ax.plot(self.train_accuracy_results.keys(), self.train_accuracy_results.values(), label='Accuracy')
        ax.set_xlabel("Epoch", fontsize=14)
        ax.legend(loc='best')
        if basename is not None:
            save_fig(fig, f'{basename}_train_metrics')
        fig.tight_layout()
        plt.show()

    def train_results(self):
        """
        Returns the training metrics stored in dictionaries.

        :return: dict: loss over epochs, dict: accuracy over epochs,
        dict: predictions (on the testing dataset) over epochs
        """

        return self.train_loss_results, self.train_accuracy_results, self.train_pred_results

    def reset_train_results(self):
        """
        Clears the training metrics.
        """
        self.train_loss_results = {}
        self.train_accuracy_results = {}
        self.train_pred_results = {}

    def get_weights(self):
        """
        Returns the model weights.

        :return: tf.tensor model weights
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def save_weights(self, path):
        """
        Saves the model weights under a specified path.

        :param str path: path where the weights are saved
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path)

    def load_weights(self, path=None):
        """
        Loads the model weights from a specified path.

        :param str path: path where the weights are saved,
        if None the weights are assumed to be saved at the checkpoints directory
        """

        if path is None:
            path = self.checkpoints_dir

        self.model.load_weights(tf.train.latest_checkpoint(path))
        logging.info(f'\tWeights loaded from {path}')

    def get_epoch_duration(self):
        """
        Measures the time for a training epoch.

        :return: float time per epoch
        """

        now = time.time()
        epoch_duration = datetime.datetime.fromtimestamp(now - self.prev_time).strftime("%M:%S.%f")[:-4]
        self.prev_time = now
        return epoch_duration

    def get_elapsed_time(self):
        """
        Measures the time since training start.

        :return: float elapsed time
        """

        return datetime.timedelta(seconds=int(time.time() - self.start_time))

    def epoch_callback(self, epoch, epoch_loss, epochs, x_val=None, y_val=None, val_freq=1000, log_freq=1000,
                       verbose=1):
        """
        Callback function, which is called after each epoch, to produce proper training logging
        and keep track of training metrics.

        :param int epoch: current epoch
        :param float epoch_loss: current loss value
        :param int epochs: number of training epochs
        :param tf.tensor x_val: input tensor of the test dataset, used to evaluate current accuracy
        :param tf.tensor y_val: output tensor of the test dataset, used to evaluate current accuracy
        :param int val_freq: number of epochs passed before trigger validation
        :param int log_freq: number of epochs passed before each logging
        """

        self.train_loss_results[epoch] = epoch_loss
        elapsed_time = self.get_elapsed_time()
        self.train_time_results[epoch] = elapsed_time

        if epoch % val_freq == 0 or epoch == 1:
            length = len(str(epochs))
            log_str = f'\tEpoch: {str(epoch).zfill(length)}/{epochs},\t' \
                      f'Loss: {epoch_loss:.4e}'

            if x_val is not None and y_val is not None:
                [mean_squared_error, errors, Y_pred] = self.evaluate(x_val, y_val)
                self.train_accuracy_results[epoch] = mean_squared_error
                self.train_pred_results[epoch] = Y_pred
                log_str += f',\tAccuracy (MSE): {mean_squared_error:.4e}'
                if mean_squared_error <= min(self.train_accuracy_results.values()):
                    self.save_weights(os.path.join(self.checkpoints_dir, 'easy_checkpoint'))

            if (epoch % log_freq == 0 or epoch == 1) and verbose == 1:
                log_str += f',\t Elapsed time: {elapsed_time} (+{self.get_epoch_duration()})'
                logging.info(log_str)

        if epoch == epochs and x_val is None and y_val is None:
            self.save_weights(os.path.join(self.checkpoints_dir, 'easy_checkpoint'))

    def evaluate(self, x_val, y_val, metric='MSE'):
        """
        Calculates the accuracy on a testing dataset.

        :param tf.tensor x_val: input tensor of the testing dataset
        :param tf.tensor y_val: output tensor of the testing dataset
        :param str metric: name of the error type, choose from 'MSE' or 'MAE'
        :return: tf.tensor mean_error: the mean squared/absolute error value,
        tf.tensor errors: the squared/absolute errors over inputs,
        tf.tensor y_pred: the prediction on the inputs of the testing dataset
        """

        y_pred = self.model.predict(x_val)
        errors = None
        if metric == 'MSE':
            errors = tf.square(y_val - y_pred)
        elif metric == 'MAE':
            errors = tf.abs(y_val - y_pred)

        mean_error = tf.reduce_mean(errors)

        return mean_error, errors, y_pred

    def prediction_time(self, batch_size, executions=1000):
        """
        Helper function to measure the mean prediction time of the neural network.

        :param int batch_size: dummy batch size of the input tensor
        :param int executions: number of performed executions to determine the mean value
        :return: float mean_prediction_time: the mean prediction time of the neural network on all executions
        """
        X = tf.random.uniform(shape=[executions, batch_size, self.input_dim], dtype=self.dtype)

        start_time = time.time()
        for execution in range(executions):
            _ = self.predict(X[execution])
        prediction_time = time.time() - start_time
        mean_prediction_time = prediction_time / executions

        return mean_prediction_time

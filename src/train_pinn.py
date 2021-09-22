import os

import click
import logging
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model.pinn import PINN
from utils.data import generate_data_points, generate_collocation_points, load_data
from utils.plotting import animate, plot_states, plot_input_sequence, plot_absolute_error
from utils.system import M_tensor, k_tensor, q_tensor, B_tensor


class ManipulatorInformedNN(PINN):
    """
    Class used to represent the Manipulator Informed Neural Network, child of PINN.
    """

    def __init__(self, layers, lb, ub, X_f=None):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bound of the inputs of the training data
        :param np.ndarray ub: upper bound of the inputs of the training data
        :param np.ndarray X_f: collocation points
        """

        super().__init__(layers, lb, ub)
        self.t = None
        self.z0 = None
        self.u = None

        if X_f is not None:
            self.set_collocation_points(X_f)

    def set_collocation_points(self, X_f):
        self.t = self.tensor(X_f[:, 0:1])
        self.z0 = self.tensor(X_f[:, 1:5])
        self.u = self.tensor(X_f[:, 5:7])

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual Physics Informed Neural Network for the approximation of the equation of motion of the
        Schunk PowerCube Serial Robot.

        :return: tf.Tensor: the prediction of the PINN
        """
        # M*D2q + k = q + B*u
        # y = [q1; q2; dq1_dt; dq2_dt]

        if X_f is None:
            t = self.t
            z0 = self.z0
            u = self.u
        else:
            t = self.tensor(X_f[:, 0:1])
            z0 = self.tensor(X_f[:, 1:5])
            u = self.tensor(X_f[:, 5:7])

        i_PR90 = tf.ones(len(t), dtype=tf.float64) * 161

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            y_pred = self.model(tf.concat([t, z0, u], axis=1))

            q1 = y_pred[:, 0:1]
            q2 = y_pred[:, 1:2]
            dq1_dt = y_pred[:, 2:3]
            dq2_dt = y_pred[:, 3:4]
            dq_dt = tf.stack([dq1_dt, dq2_dt], axis=1)

        dq1_dt_tf = tape.gradient(q1, t)[:, 0]
        dq2_dt_tf = tape.gradient(q2, t)[:, 0]
        dq_dt_tf = tf.stack([dq1_dt_tf, dq2_dt_tf], axis=1)

        d2q1_dt_tf = tape.gradient(dq1_dt, t)[:, 0]
        d2q2_dt_tf = tape.gradient(dq2_dt, t)[:, 0]
        d2q_dt_tf = tf.stack([d2q1_dt_tf, d2q2_dt_tf], axis=1)

        M_tf = M_tensor(q2[:, 0], i_PR90)
        k_tf = k_tensor(dq1_dt_tf, q2[:, 0], dq2_dt_tf)
        q_tf = q_tensor(q1[:, 0], dq1_dt_tf, q2[:, 0], dq2_dt_tf)
        B_tf = B_tensor(i_PR90)

        f_pred = tf.concat([dq_dt_tf - dq_dt[:, :, 0],
                            tf.linalg.matvec(M_tf, d2q_dt_tf) + k_tf - q_tf - tf.linalg.matvec(B_tf, u)], axis=1)

        return f_pred


if __name__ == "__main__":
    LOAD_WEIGHTS = False
    TRAIN_NET = True

    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False  # don't setup fonts from rc parameters
    })

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logging.info("TensorFlow version: {}".format(tf.version.VERSION))
    logging.info("Eager execution: {}".format(tf.executing_eagerly()))

    # Hyper parameter
    N_train = 2
    epochs = 500000
    N_phys = 20000
    N_data = 100

    logging.info(f'Epochs: {epochs}')
    logging.info(f'N_data: {N_data}')
    logging.info(f'N_phys: {N_phys}')

    # Paths
    data_path = os.path.join('../resources/data.npz')
    weights_path = os.path.join('../resources/weights')

    lb, ub, input_dim, output_dim, X_test, Y_test, X_star, Y_star = load_data(data_path)

    N_layer = 4
    N_neurons = 64
    layers = [input_dim, *N_layer * [N_neurons], output_dim]

    # PINN initialization
    pinn = ManipulatorInformedNN(layers, lb, ub)

    if LOAD_WEIGHTS:
        pinn.load_weights(weights_path)

    # PINN training
    if TRAIN_NET:
        for i in range(N_train):
            # Generate training data via LHS
            X_data, Y_data = generate_data_points(N_data, lb, ub)
            X_phys = generate_collocation_points(N_phys, lb, ub)
            pinn.set_collocation_points(X_phys)
            logging.info(f'\t{i + 1}/{N_train} Start training of the PINN')
            start_time = time.time()
            pinn.fit(X_data, Y_data, epochs, X_star, Y_star, optimizer='lbfgs', learning_rate=1,
                     val_freq=1000, log_freq=1000)

    # PINN Evaluation
    Y_pred, F_pred = pinn.predict(X_test)

    t_step = X_test[1, 0] - X_test[0, 0]
    tau = 0.2
    T = np.arange(t_step, 20 * tau + t_step, t_step)

    plot_input_sequence(T, X_test[:, 5:])
    plot_states(T, Y_test, Y_pred)
    plot_absolute_error(T, Y_test, Y_pred)

    animate(Y_test[::10], [Y_pred[::10]], ['PINN'], fps=1 / (10 * t_step), save_ani=False)

    if click.confirm('Do you want to save (overwrite) the models weights?'):
        pinn.save_weights(os.path.join(weights_path, 'easy_checkpoint'))

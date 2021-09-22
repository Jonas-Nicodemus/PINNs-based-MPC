import tensorflow as tf

from scipy.constants import g


@tf.function
def f(t, x, u):
    M_tf = M(x[1])
    k_tf = k(x[2], x[1], x[3])
    q_tf = q(x[0], x[2], x[1], x[3])
    B_tf = B()

    dx12dt = x[2:]

    dx34dt = tf.linalg.solve(M_tf, tf.expand_dims(-k_tf + q_tf + tf.linalg.matvec(B_tf, u), 1))[:, 0]

    dxdt = tf.concat((dx12dt, dx34dt), 0)

    return dxdt


def M(beta, i_PR90=161.):
    """
    Returns mass matrix of the robot for beta.

    :param tf.Tensor beta: tensor from beta value
    :param float i_PR90: motor constant
    :return: tf.Tensor M_tf: mass matrx of the robot
    """
    M_1 = tf.stack([0.00005267 * i_PR90 ** 2 + 0.6215099724 * tf.cos(beta) + 0.9560375168565, 0.00005267 * i_PR90 +
                    0.3107549862 * tf.cos(beta) + 0.6608899068565], axis=0)

    M_2 = tf.stack([0.00005267 * i_PR90 + 0.3107549862 * tf.cos(beta) + 0.6608899068565,
                    0.00005267 * i_PR90 ** 2 + 0.6608899068565], axis=0)

    M_tf = tf.stack([M_1, M_2], axis=1)

    return M_tf


def k(dalpha_dt, beta, dbeta_dt):
    """
    Returns stiffness vector of the robot for a set of generalized coordinates.

    :param tf.Tensor dalpha_dt: tensor from values of the first derivation of alpha
    :param tf.Tensor beta: tensor from beta values
    :param tf.Tensor dbeta_dt: tensor from values of the first derivation of beta
    :return: tf.Tensor: stiffness vector of the robot
    """

    return tf.stack([0.040968 * dalpha_dt ** 2 * tf.sin(beta) * (0.18 * tf.cos(beta) + 0.5586) - 0.18 * tf.sin(beta) *
                     (1.714 * (0.07205 * dalpha_dt + 0.07205 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 1.714 *
                      (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 1.714 *
                      (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 1.714 *
                      (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 0.30852 *
                      dalpha_dt ** 2 * tf.cos(beta) + 1.714 * (0.0574 * dalpha_dt + 0.0574 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt) + 1.714 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt)) -
                     0.36 * tf.sin(beta) *
                     (0.1138 * (0.06415 * dalpha_dt + 0.06415 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.1138 * (0.07205 * dalpha_dt + 0.07205 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.1138 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.1138 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.1138 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.020484 * dalpha_dt ** 2 * tf.cos(beta) + 0.1138 * (0.0574 * dalpha_dt + 0.0574 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt) + 0.1138 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt) + 0.1138 * (0.03 * dalpha_dt + 0.03 * dbeta_dt) * (dalpha_dt + dbeta_dt)) -
                     0.18 * tf.sin(beta) *
                     (2.751 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 0.49518 *
                      dalpha_dt ** 2 * tf.cos(beta)) - 0.18 * tf.sin(beta) *
                     (1.531 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 1.531 *
                      (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 0.27558 * dalpha_dt ** 2 *
                      tf.cos(beta) + 1.531 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)) -
                     0.18 * tf.sin(beta) *
                     (0.934 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.934 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.934 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.16812 * dalpha_dt ** 2 * tf.cos(beta) + 0.934 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt)) +
                     0.16812 * dalpha_dt ** 2 * tf.sin(beta) * (
                             0.18 * tf.cos(beta) + 0.335) + 0.49518 * dalpha_dt ** 2 *
                     tf.sin(beta) * (0.18 * tf.cos(beta) + 0.04321) + 0.30852 * dalpha_dt ** 2 * tf.sin(beta) *
                     (0.18 * tf.cos(beta) + 0.46445) + 0.27558 * dalpha_dt ** 2 * tf.sin(beta) * (0.18 * tf.cos(beta)
                                                                                                  + 0.24262),
                     0.3107549862 * dalpha_dt ** 2 * tf.sin(beta)], axis=0)


def q(alpha, dalpha_dt, beta, dbeta_dt):
    """
    Returns reaction forces vector of the robot for a set of generalized coordinates.

    :param tf.Tensor alpha: tensor from alpha values
    :param tf.Tensor dalpha_dt: tensor from values of the first derivation of alpha
    :param tf.Tensor beta: tensor from beta values
    :param tf.Tensor dbeta_dt: tensor from values of the first derivation of beta
    :return: tf.Tensor: reaction forces vectors of the robot
    """

    return tf.stack(
        [0.33777 * g * tf.sin(alpha) - 3.924 * tf.tanh(5 * dalpha_dt) - 10.838 * tf.tanh(10 * dalpha_dt) -
         2.236 * tf.tanh(20 * dalpha_dt) - 76.556 * dalpha_dt - 1.288368 * g * tf.cos(alpha + beta) *
         tf.sin(beta) + 0.2276 * g * tf.sin(alpha + beta) * (0.18 * tf.cos(beta) + 0.5586) +
         0.934 * g * tf.sin(alpha + beta) * (0.18 * tf.cos(beta) + 0.335) + 2.751 * g *
         tf.sin(alpha + beta) * (0.18 * tf.cos(beta) + 0.04321) + 1.714 * g * tf.sin(alpha + beta) *
         (0.18 * tf.cos(beta) + 0.46445) + 1.531 * g * tf.sin(alpha + beta) * (0.18 * tf.cos(beta) +
                                                                               0.24262),
         1.72641659 * g * tf.sin(alpha + beta) - 0.368 * tf.tanh(5 * dbeta_dt) -
         0.368 * tf.tanh(10 * dbeta_dt) - 8.342 * tf.tanh(100 * dbeta_dt) -
         0.492 * tf.sign(dbeta_dt) - 56.231 * dbeta_dt], axis=0)


def B(i_PR90=161.):
    """
    Returns input matrix of the robot.

    :param float i_PR90: constant
    :return: tf.Tensor: input matrix of the robot
    """
    i_PR90 = tf.convert_to_tensor(i_PR90, dtype=tf.float64)

    B_1 = tf.stack([i_PR90, 0.0], axis=0)

    B_2 = tf.stack([0.0, i_PR90], axis=0)

    B_tf = tf.stack([B_1, B_2], axis=1)

    return B_tf

def M_tensor(beta, i_PR90):
    """
    Returns mass matrices of the robot for multiple values for beta.

    :param tf.Tensor beta: tensor from beta values
    :param float i_PR90: constant
    :return: tf.Tensor M_tf: mass matrices of the robot
    """
    M_1 = tf.stack([0.00005267 * i_PR90 ** 2 + 0.6215099724 * tf.cos(beta) + 0.9560375168565, 0.00005267 * i_PR90 +
                    0.3107549862 * tf.cos(beta) + 0.6608899068565], axis=1)

    M_2 = tf.stack([0.00005267 * i_PR90 + 0.3107549862 * tf.cos(beta) + 0.6608899068565,
                    0.00005267 * i_PR90 ** 2 + 0.6608899068565], axis=1)

    M_tf = tf.stack([M_1, M_2], axis=2)

    return M_tf


def k_tensor(dalpha_dt, beta, dbeta_dt):
    """
    Returns stiffness vectors of the robot for multiple values of generalized coordinates.

    :param tf.Tensor dalpha_dt: tensor from values of the first derivation of alpha
    :param tf.Tensor beta: tensor from beta values
    :param tf.Tensor dbeta_dt: tensor from values of the first derivation of beta
    :return: tf.Tensor: stiffness vectors of the robot
    """

    return tf.stack([0.040968 * dalpha_dt ** 2 * tf.sin(beta) * (0.18 * tf.cos(beta) + 0.5586) - 0.18 * tf.sin(beta) *
                     (1.714 * (0.07205 * dalpha_dt + 0.07205 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 1.714 *
                      (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 1.714 *
                      (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 1.714 *
                      (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 0.30852 *
                      dalpha_dt ** 2 * tf.cos(beta) + 1.714 * (0.0574 * dalpha_dt + 0.0574 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt) + 1.714 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt)) -
                     0.36 * tf.sin(beta) *
                     (0.1138 * (0.06415 * dalpha_dt + 0.06415 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.1138 * (0.07205 * dalpha_dt + 0.07205 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.1138 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.1138 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.1138 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.020484 * dalpha_dt ** 2 * tf.cos(beta) + 0.1138 * (0.0574 * dalpha_dt + 0.0574 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt) + 0.1138 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt) + 0.1138 * (0.03 * dalpha_dt + 0.03 * dbeta_dt) * (dalpha_dt + dbeta_dt)) -
                     0.18 * tf.sin(beta) *
                     (2.751 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 0.49518 *
                      dalpha_dt ** 2 * tf.cos(beta)) - 0.18 * tf.sin(beta) *
                     (1.531 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 1.531 *
                      (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) + 0.27558 * dalpha_dt ** 2 *
                      tf.cos(beta) + 1.531 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)) -
                     0.18 * tf.sin(beta) *
                     (0.934 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.934 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.934 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt) +
                      0.16812 * dalpha_dt ** 2 * tf.cos(beta) + 0.934 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) *
                      (dalpha_dt + dbeta_dt)) +
                     0.16812 * dalpha_dt ** 2 * tf.sin(beta) * (
                             0.18 * tf.cos(beta) + 0.335) + 0.49518 * dalpha_dt ** 2 *
                     tf.sin(beta) * (0.18 * tf.cos(beta) + 0.04321) + 0.30852 * dalpha_dt ** 2 * tf.sin(beta) *
                     (0.18 * tf.cos(beta) + 0.46445) + 0.27558 * dalpha_dt ** 2 * tf.sin(beta) * (0.18 * tf.cos(beta)
                                                                                                  + 0.24262),
                     0.3107549862 * dalpha_dt ** 2 * tf.sin(beta)], axis=1)


def q_tensor(alpha, dalpha_dt, beta, dbeta_dt):
    """
    Returns reaction forces vectors of the robot for multiple values of generalized coordinates.

    :param tf.Tensor alpha: tensor from alpha values
    :param tf.Tensor dalpha_dt: tensor from values of the first derivation of alpha
    :param tf.Tensor beta: tensor from beta values
    :param tf.Tensor dbeta_dt: tensor from values of the first derivation of beta
    :return: tf.Tensor: reaction forces vectors of the robot
    """

    return tf.stack(
        [0.33777 * g * tf.sin(alpha) - 3.924 * tf.tanh(5 * dalpha_dt) - 10.838 * tf.tanh(10 * dalpha_dt) -
         2.236 * tf.tanh(20 * dalpha_dt) - 76.556 * dalpha_dt - 1.288368 * g * tf.cos(alpha + beta) *
         tf.sin(beta) + 0.2276 * g * tf.sin(alpha + beta) * (0.18 * tf.cos(beta) + 0.5586) +
         0.934 * g * tf.sin(alpha + beta) * (0.18 * tf.cos(beta) + 0.335) + 2.751 * g *
         tf.sin(alpha + beta) * (0.18 * tf.cos(beta) + 0.04321) + 1.714 * g * tf.sin(alpha + beta) *
         (0.18 * tf.cos(beta) + 0.46445) + 1.531 * g * tf.sin(alpha + beta) * (0.18 * tf.cos(beta) +
                                                                               0.24262),
         1.72641659 * g * tf.sin(alpha + beta) - 0.368 * tf.tanh(5 * dbeta_dt) -
         0.368 * tf.tanh(10 * dbeta_dt) - 8.342 * tf.tanh(100 * dbeta_dt) -
         0.492 * tf.sign(dbeta_dt) - 56.231 * dbeta_dt], axis=1)


def B_tensor(i_PR90):
    """
    Returns input matrices of the robot.

    :param float i_PR90: constant
    :return: tf.Tensor: input matrices of the robot
    """
    B_1 = tf.stack([i_PR90, tf.zeros(i_PR90.shape, dtype=tf.float64)], axis=1)

    B_2 = tf.stack([tf.zeros(i_PR90.shape, dtype=tf.float64), i_PR90], axis=1)

    B_tf = tf.stack([B_1, B_2], axis=2)

    return B_tf

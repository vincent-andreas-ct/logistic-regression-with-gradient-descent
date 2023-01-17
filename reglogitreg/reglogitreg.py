# import dependencies 
import numpy as np

def sigmoid(z):
    """
    Apply sigmoid function for the given x
    Args:
    z (ndarray(m,) or scalar)   : value(s) to be used for sigmoid function

    Returns:
    sig_z: (ndarray(m,) or scalar): result of sigmoid function from the given x
    """
    min_z = z * -1
    exp_z = np.exp(min_z)
    den_sig = 1 + exp_z
    sig_z = 1 / den_sig
    return sig_z

def calc_cost_reg(x, y, w, b, lambda_):
    """
    Calculate cost function for the given x with regularization
    Args:
    x (ndarray(m, n))   : feature(s) or variable(s) to used for calculating cost
    y (ndarray(m,))     : target values
    w (ndarray(n,))     : coefficient(s) to used for calculating cost
    b (scalar)          : intercept - parameter
    lambda_ (scalar)    : controls the amount of regularization

    Returns:
    cost (scalar)       : cost, with the given w and b
    """
    m, n = x.shape
    z_pred = np.matmul(x, w.reshape(-1, 1)) + b
    sig_z = sigmoid(z_pred.flatten())
    left_part = (-1 * y) * np.log(sig_z)
    right_part = (1 - y) * np.log(1- sig_z)
    loss= left_part - right_part
    temp_cost = np.sum(loss)/m

    # calculate regularization cost
    reg_cost = 0.
    for col in range(n):
        reg_cost += (w[col] ** 2)
    cost = temp_cost + ((lambda_/(2 * m)) * reg_cost)
    return cost

def calc_gradient_reg(x, y, w, b, lambda_):
    """
    Calculate the gradient with using regularization
    Args:
    x (ndarray(m, n))   : feature(s) or variable(s) to used for calculating gradient
    y (ndarray(m,))     : target values
    w (ndarray(n,))     : coefficient(s) to used for calculating cost
    b (scalar)          : intercept - parameter
    lambda_ (scalar)    : controls the amount of regularization

    Returns:
    g_coeffs (ndarray(n,))  : gradient of the cost with respect to the parameter w
    g_intercept (scalar)    : gradient of the cost with respect to the parameter b
    """
    m, n = x.shape
    g_coeffs = np.zeros((n,))
    z_pred = np.matmul(x, w.reshape(-1, 1)) + b
    sig_z = sigmoid(z_pred)
    diff = sig_z.flatten() - y
    for col in range(n):
        temp = np.matmul(diff.reshape(1, -1), x[:, col].reshape(-1, 1)).item(0)
        reg = (lambda_/m) * w[col]
        g_coeffs[col] = (temp/m) + reg
    g_intercept = np.sum(diff)/m
    return g_coeffs, g_intercept

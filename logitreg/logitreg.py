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

def calc_cost(x, y, w, b, lambda_=None):
    """
    Calculate loss function for logistic regression with the given x, y
    Args:
    x (ndarray(m,) or ndarray(m,n)) : value(s) to be used for calculating the loss
    y (ndarray(m,))                 : expected output from sigmoid(x)
    w (ndarray(n,) or scalar)       : coefficient(s) - parameter(s)
    b (scalar)                      : intercept - parameter
    lambda_ (unused placeholder)

    Returns:
    loss (ndarray(m,))              : losses calculated with the given x and y
    """
    try:
        m, n = x.shape
        z_pred = np.matmul(x, w.reshape(-1, 1)) + b
    except Exception as e:
        m = x.shape[0]
        z_pred = (x * w) + b
    sig_z = sigmoid(z_pred.flatten())
    left_part = (-1 * y) * np.log(sig_z)
    right_part = (1 - y) * np.log(1 - sig_z)
    loss = left_part - right_part
    cost = np.sum(loss)/m
    return cost

def calc_gradient(x, y, w, b, lambda_=None):
    """
    Calculate gradient from the given w and b for values of x
    Args:
    x (ndarray(m,) or ndarray(m,n)) : value(s) to be used for calculating the loss
    y (ndarray(m,))                 : expected output from sigmoid(x)
    w (ndarray(n,) or scalar)       : coefficient(s) - parameter(s)
    b (scalar)                      : intercept - parameter
    lambda_ (ununsed placeholder)

    Returns
    g_coeffs (ndarray(n,) or scalar): gradient for coefficient(s)
    g_intercept (scalar)            : gradient for intercept
    """
    try:
        m, n = x.shape
        g_coeffs = np.zeros((n,))
        z_pred = np.matmul(x, w.reshape(-1, 1)) + b
        sig_z = sigmoid(z_pred)
        diff = sig_z.flatten() - y
        for col in range(n):
            temp = np.matmul(diff.reshape(1, -1),
                             x[:, col].reshape(-1, 1)).item(0)
            g_coeffs[col] = temp/m
        g_intercept = np.sum(diff)/m
    except Exception as e:
        m = x.shape[0]
        z_pred = (x * w) + b
        sig_z = sigmoid(z_pred)
        diff = sig_z.flatten() - y
        temp = np.matmul(diff.reshape(1, -1), x.reshape(-1, 1)).item(0)
        g_coeffs = temp/m
        g_intercept = np.sum(diff)/m
    return g_coeffs, g_intercept
    

def logreg_gd(x, y, params, cost_func, grad_func):
    """
    Perform gradient descent to find the best parameters for logistic regression model
    Args:
    x (ndarray(m,) or ndarray(m,n)) : value(s) to be used in gradient descent
    y (ndarray(m,))                 : expected output from sigmoid(x)
    params (dict) must contains the following:
    iters: int                  -> number of iterations to perform gradient descent
    alpha: scalar               -> learning rate for gradient descent
    w_in (ndarray(n,) or scalar)-> coefficient(s) - parameter(s)
    b_in (scalar)               -> intercept - parameter

    Returns:
    w_conv (ndarray(n,) or scalar)  : coefficient(s) after performing iters gradient descent
    b_conv                          : intercept after performing iters gradient descent
    J_history                       : history of cost J(w,b) of the given x
    """
    w_conv = params['w_in']
    b_conv = params['b_in']
    intervals = np.ceil(params['iters']/5)
    if params['iters']<100000:
        J_history = np.zeros((params['iters'],))
    else:
        J_history = np.zeros((100000,))
    for idx in range(params['iters']):
        g_coeffs, g_intercept = grad_func(x, y, w_conv, b_conv, params['lambda_'])
        cost = cost_func(x, y, w_conv, b_conv, params['lambda_'])
        if (idx%intervals) == 0:
            print("Iteration {}: \nCost {}".format(idx, cost))
        if idx<=100000:
            J_history[idx] = cost
        w_conv = w_conv - params['alpha'] * g_coeffs
        b_conv = b_conv - params['alpha'] * g_intercept
    print("Iteration {}: \nCost {}".format(idx+1, cost))
    return w_conv, b_conv, J_history

def predict(x, w, b):
    """
    Predict binary label (0 or 1) of the given x using parameters (w and b)
    Args:
    x (ndarray(m,) or ndarray(m,n)) : value(s) to be used for prediction
    w (scalar or ndarray(n,))       : coefficient(s) - parameter(s)
    b (scalar)                      : intercept - parameter

    Returns:
    p_y (ndarray(m,))               : predictions for x using threshold at 0.5
    """
    try:
        m, n = x.shape
    except Exception as e:
        m = x.shape[0]
    p_y = np.zeros((m,))
    for row in range(m):
        sig_z = sigmoid(np.dot(x[row], w) + b)
        if sig_z >= 0.5:
            p_y[row] = 1
    return p_y

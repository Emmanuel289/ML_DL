
"""
Given m inputs (features) x, m corresponding outputs (targets), y and a machine learning
model f given by the equation y = f(x, w, b) = wx + b where w and b are the weight and bias, 
respectively,
The cost function in computing estimates y^ is given by 
J(w, b) = (1 /(2 m)) * sum_(1, m) * (f(x, w, b) - y)^2
The equations for updating parameters using gradient descent are:
w = w - alpha * dJ/dw = w - alpha * (1 / m) * sum_(1, m)x(f(w, b) - y)  --- (i)
b = b - alpha * dJ/db  = w - alpha * (1 / m) * sum_(1, m) (f(w, b) - y)  --- (ii)
where alpha is the learning rate. The updates in w and b are repeated until convergence
"""

# Example of predicting the price of a house using gradient descent

import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process

# load our data set
x_train = np.array([1.0, 2.0])  # features representing sq ft area in 1000s
# target values representing prices in 1000 dollars
y_train = np.array([300.0, 500.0])

print(x_train.shape[0])


# Compute cost: J(w, b) = (1 /(2 m)) * sum_(1, m) * (f(w, b) - y)^2

def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2

    total_cost = 1 / (2 * m) * cost

    return total_cost


# Compute gradients: dJ/dw = 1/m * sum_(1, m) * x (f(w, b) - y), dJ/db = 1/m * sum_(1, m) * (f(w, b) - y)


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
    x (nd array (m,)): Data, m examples
    y (nd array (m,)): target values
    w, b (scalar): model parameters

    Returns:
    dj_dw (scalar): The graJ(w, b) = (1 /(2 m)) * sum_(1, m) * (f(x, w, b) - y)^2dient of the cost w.r.t parameters w
    dj_db (scalar): The gradient of the cost w.r.t parameters b
    """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


# Plot 1: Cost vs w

w = np.arange(0, 450, 50)  # weights
b = 100  # fixed bias
costs = [compute_cost(x_train, y_train, w[i], b) for i in range(w.shape[0])]


plt.plot(w, costs, scalex=1000, scaley=10)


# plot figure
def plot_cost_vs_w():
    plt.figure(1)
    plt.title('Cost vs w, with gradient; b set to 100')
    plt.ylabel('Cost')
    plt.xlabel('w')
    plt.show()


# # Run plot in a separate child process
# proc = Process(target=plot_cost_vs_w)
# proc.start()
# proc.join(3)  # Wait for 3 seconds to visualize plot
# proc.terminate()


# Gradient descent


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # return w and J,w history for graphing


# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()


# Predict prices using the final w and b values

print(
    f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(
    f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(
    f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")


# Plot gradient descent progress over number of iterations
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

w_vals = [elem[0] for elem in p_hist]
b_vals = [elem[1] for elem in p_hist]

# Assuming J_hist is a 2D array representing the values of the cost function
J_hist = np.array([J_hist], [J_hist])

# Create a grid of points
w_grid, b_grid = np.meshgrid(w_vals, b_vals)

# Plot the contour
contour = ax.contour(w_grid, b_grid, J_hist, cmap='viridis')

# Add labels and a colorbar
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_title('Contour Plot')
cbar = fig.colorbar(contour)

# Show the plot
plt.show()

import numpy as np

def mean_squared_error(estimates, targets):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is
    estimated.

    Refer to the slides, or read more here:
      https://en.wikipedia.org/wiki/Mean_squared_error

    Do not import or use these packages: scipy, sklearn, sys, importlib.

    Args:
        estimates (np.ndarray): the estimated values (should be the same shape as targets)
        targets (np.ndarray): the ground truth values

    Returns:
        MSE (float): the mean squared error across all estimates and targets
    """

    # print(f"estimates = {estimates}")
    # print(f"targets = {targets}")
    # print(f"shape = {estimates.shape}")

    MSE = 0.0

    for i in range(0,estimates.shape[0]):
        MSE += (targets[i] - estimates[i])**2

    # raise NotImplementedError

    MSE = (1/estimates.shape[0])*MSE

    # print(f"shape = {MSE}")

    return float(MSE)

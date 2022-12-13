from xml.dom import NotFoundErr
import numpy as np


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """

    # print(f"X = {X}")
    # print(f"Y = {Y}")
    # print(f"shape X = {X.shape}")
    # print(f"shape Y = {Y.shape}")

    # D = np.zeros((X.shape[0],X.shape[1]))

    # for i in range(0, X.shape[0]):
    #     for j in range(0, X.shape[1]):
            #D[i,j] = np.linalg.norm(X[i,j] - Y[i,j])
            # D[i,j] = (abs(Y[i,j] - X[i,j]))
            # D[i,j] = np.sqrt((Y[i,j] - X[i,j])**2)

    D = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)



    # print(f"D = {D}")
    # print(f"shape D = {D.shape}")


    #raise NotImplementedError

    return D


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """

    D =  np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=-1)

    # print(f"D = {D}")
    # print(f"shape D = {D.shape}")

    return D
    #raise NotImplementedError


def cosine_distances(X, Y):
    """Compute pairwise Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """

    # print(f"X = {X}")
    # print(f"Y = {Y}")
    # print(f"shape X = {X.shape}")
    # print(f"shape Y = {Y.shape}")

    D = np.dot(X,Y.T)/(np.linalg.norm(Y[None, :, :], axis=-1)*np.linalg.norm(X[:, None, :], axis=-1))

    # print(f"D = {D}")
    # print(f"shape D = {D.shape}")

    ONE = np.ones((X.shape[0],Y.shape[0]))

    # print(f"D = {D}")
    # print(f"shape D = {D.shape}")

    return ONE - D

    # raise NotImplementedError

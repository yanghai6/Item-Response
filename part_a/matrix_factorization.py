from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    u[n] += lr * (c - u[n].T.dot(z[q])) * z[q]
    z[q] += lr * (c - u[n].T.dot(z[q])) * u[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix, u, z
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    us = []
    zs = []
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        us.append(u)
        zs.append(z)
    miu = sum(train_data['is_correct']) / len(train_data['is_correct'])
    mat = miu + u.dot(z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, us, zs


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # print('===== SVD =====')
    # ks = [1, 4, 9, 21, 51, 101]
    # valid_acc = []
    # for k in ks:
    #     transformed_mat = svd_reconstruct(train_matrix, k)
    #     valid_acc.append(sparse_matrix_evaluate(val_data, transformed_mat, threshold=0.5))
    # k_star = ks[valid_acc.index(max(valid_acc))]
    # test_acc = sparse_matrix_evaluate(test_data, svd_reconstruct(train_matrix, k_star), threshold=0.5)
    # print(f'k* = {k_star}, validation accuracy: {max(valid_acc)}, test accuracy: {test_acc}')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    print('===== ALS =====')
    ks = [1, 4, 9, 21, 51, 101]
    lr = 0.01
    num_iteration = 100
    valid_acc = []
    us = []
    zs = []
    for k in range(5, 10):
        transformed_mat, u, z = als(train_data, k, lr, num_iteration)
        valid_acc.append(sparse_matrix_evaluate(val_data, transformed_mat, threshold=0.5))
        us.append(u)
        zs.append(z)
    print(valid_acc)
    k_star = ks[valid_acc.index(max(valid_acc))]
    test_acc = sparse_matrix_evaluate(test_data, svd_reconstruct(train_matrix, k_star), threshold=0.5)
    print(f'k* = {k_star}, validation accuracy: {max(valid_acc)}, test accuracy: {test_acc}')

    train_losses = []
    val_losses = []
    for i in range(num_iteration):
        u = us[valid_acc.index(max(valid_acc))][i]
        z = zs[valid_acc.index(max(valid_acc))][i]
        train_losses.append(squared_error_loss(train_data, u, z))
        val_losses.append(squared_error_loss(val_data, u, z))
    plt.plot([i for i in range(len(train_losses))], train_losses, label='train')
    plt.plot([i for i in range(len(val_losses))], val_losses, label='validation')
    plt.xlabel('iteration')
    plt.ylabel('square error loss')
    plt.title('validation accuracy item based knn')
    plt.savefig("Q3 (e)")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

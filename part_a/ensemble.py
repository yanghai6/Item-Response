# TODO: complete this file.
from utils import *
from part_a import item_response

import numpy as np
from sklearn.utils import resample


def bootstrap(data, val_data, size, iterations):
    """ Bootstrapping the data to get a sample and use irt to get a model.

    :param data: the sparse matrix of the training set
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param size: size of the bootstrap sample
    :iterations: number of iterations in the irt algorithm
    :return: beta, theta from irt
    """
    sample = resample(data, n_samples=size, replace=True)
    theta, beta, train_nllk_lst, valid_nllk_lst = \
        item_response.irt(sample, val_data, 0.001, iterations)
    return beta, theta


def evaluate(data, betas, thetas):
    """ Evaluate the ensemble model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param betas: List of vectors
    :param theta: List of vectors
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        p_a = 0
        for j in range(len(betas)):
            beta = betas[j]
            theta = thetas[j]
            x = (theta[u] - beta[q]).sum()
            p_a += item_response.sigmoid(x)
        p_a = p_a / len(betas)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    # import data
    train_data = load_train_csv("../data")
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    # train 3 models based on bootstrapping samples
    size = sparse_matrix.shape[0]
    iterations = 100
    betas = []
    thetas = []
    for i in range(3):
        beta, theta = bootstrap(sparse_matrix, val_data, size, iterations)
        betas.append(beta)
        thetas.append(theta)
    # ensemble
    valid_acc = evaluate(val_data, betas, thetas)
    test_acc = evaluate(test_data, betas, thetas)
    print("Validation Accuracy:", valid_acc)
    print("Test Accuracy:", test_acc)


if __name__ == "__main__":
    main()

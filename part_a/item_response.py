from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list} -> sparse matrix instead
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    diff = theta.reshape((data.shape[0], 1)) - beta.reshape((1, data.shape[1]))
    m = np.multiply(data, diff) - np.log(1 + np.exp(diff))
    m[np.isnan(m)] = 0
    log_lklihood = np.sum(m)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list} -> sparse matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    diff = theta.reshape((data.shape[0], 1)) - beta.reshape((1, data.shape[1]))
    # gradient descent of theta
    m1 = data - sigmoid(diff)
    m1[np.isnan(m1)] = 0
    d1 = np.sum(m1, axis=1)
    theta = theta + lr * d1
    # gradient descent of beta
    m2 = -data + sigmoid(diff)
    m2[np.isnan(m2)] = 0
    d2 = np.sum(m2, axis=0)
    beta = beta + lr * d2
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst, train_nllk_lst, valid_nllk_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(data.shape[0])
    beta = np.zeros(data.shape[1])

    # convert val_data to the sparse matrix form
    val_matrix = np.nan * np.empty(data.shape)
    for i in range(len(val_data['user_id'])):
        u = val_data['user_id'][i]
        q = val_data['question_id'][i]
        val_matrix[u, q] = val_data['is_correct'][i]

    # list to stored values
    val_acc_lst = []
    train_nllk_lst = []
    valid_nllk_lst = []

    for i in range(iterations):
        # log-likelihood of training set
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_nllk_lst.append(-train_neg_lld)
        # log-likelihood of validation set
        valid_neg_lld = neg_log_likelihood(val_matrix, theta=theta, beta=beta)
        valid_nllk_lst.append(-valid_neg_lld)
        # validation accuracy
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(train_neg_lld, score))
        # update parameters
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_nllk_lst, valid_nllk_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # tune the hyperparameters
    iterations = 100
    theta, beta, train_nllk_lst, valid_nllk_lst = \
        irt(sparse_matrix, val_data, 0.001, iterations)
    # plot the log-likelihood curve
    plt.figure()
    plt.title("Train-Loglikelihood of Each Iterations")
    plt.ylabel("Loglikelihood")
    plt.xlabel("Iterations")
    plt.plot(range(iterations), train_nllk_lst, label='train')
    plt.legend()
    plt.savefig('train_loglikelihood.png')
    plt.figure()
    plt.title("Validation Loglikelihood of Each Iterations")
    plt.ylabel("Loglikelihood")
    plt.xlabel("Iterations")
    plt.plot(range(iterations), valid_nllk_lst, label='validation')
    plt.legend()
    plt.savefig('validation_loglikelihood.png')
    # report the valid and test accuracy
    print("lr:", 0.001)
    print("iterations:", iterations)
    valid_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Validation Accuracy:", valid_acc)
    print("Test Accuracy:", test_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    plt.figure()
    plt.title('Probability of the Correct Response')
    plt.xlabel('theta')
    plt.ylabel('probability')
    for i in range(5):
        b = beta[i]
        p_lst = []
        theta_sort = np.sort(theta)
        for t in theta_sort:
            p = sigmoid(t - b)
            p_lst.append(p)
        plt.plot(theta_sort, p_lst, label='question' + str(i))
    plt.legend()
    plt.savefig('probability.png')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

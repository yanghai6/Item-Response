from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.transpose()).transpose()
    acc = sparse_matrix_evaluate(valid_data, mat)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    ks = [1, 6, 11, 16, 21, 26]

    print('k-nn based on student similarity')
    print('Validation Accuracy')
    valid_acc = []
    for k in ks:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        valid_acc.append(acc)
        print(f'k = {k}: {acc}')
    plt.plot([k for k in ks], valid_acc)
    plt.scatter([k for k in ks], valid_acc)
    plt.xlabel('k value')
    plt.ylabel('accuracy')
    plt.title('validation accuracy student based knn')
    plt.savefig("Q1 (a)")
    # pick k*
    k_star = ks[valid_acc.index(max(valid_acc))]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print('Test accuracy with chosen k*')
    print(f'k* = {k_star}: {test_acc}')

    print('==================')

    print('k-nn based on item similarity')
    print('Validation Accuracy')
    valid_acc = []
    for k in ks:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        valid_acc.append(acc)
        print(f'k = {k}: {acc}')
    plt.plot([k for k in ks], valid_acc)
    plt.scatter([k for k in ks], valid_acc)
    plt.xlabel('k value')
    plt.ylabel('accuracy')
    plt.title('validation accuracy item based knn')
    plt.savefig("Q1 (c)")
    # pick k*
    k_star = ks[valid_acc.index(max(valid_acc))]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print('Test accuracy with chosen k*')
    print(f'k* = {k_star}: {test_acc}')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

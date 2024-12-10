import sys
from einops import rearrange
import numpy as np
from typing import Protocol
from scipy.optimize import linear_sum_assignment
from scipy.special import kl_div
from torch import tensor
import torch  # type: ignore

np.set_printoptions(precision=2, suppress=True)

class FeaturesMetric(Protocol):
    def __call__(self, y_ND: np.ndarray, x_ND: np.ndarray) -> float: ...


def perfect(N: int = 50, D: int = 10):
    x_ND = (np.random.random((N, D)) > 0.1).astype(int)

    # permute the features "randomly"
    # we should be able to match all features perfectly

    perm = np.random.permutation(D)
    y_ND = x_ND[:, perm]
    inv_perm = np.argsort(perm)
    assert np.all(y_ND[:, inv_perm] == x_ND)

    return x_ND, y_ND


# def random(N: int = 50, D: int = 10):
#     # random features
#     x_ND = np.random.random((N, D))
#     y_ND = np.random.random((N, D))
#     return x_ND, y_ND

def random_sparse(N: int = 50, D: int = 10):
    x_ND = (np.random.random((N, D)) > 0.1).astype(int)
    y_ND = (np.random.random((N, D)) > 0.1).astype(int)
    return x_ND, y_ND


class FeatureErrorMetric(Protocol):
    def __call__(self, x_ND: np.ndarray, y_ND: np.ndarray) -> np.ndarray:
        """
        Compute the error between each pair of features.
        """
        ...


class MatrixScorer(Protocol):
    def __call__(self, pairwise_features_mse_DD: np.ndarray) -> float: ...

def closeness(pairwise_features_mse_DD: np.ndarray) -> float:
    return np.mean(pairwise_features_mse_DD)

def permutation_matching(pairwise_features_mse_DD: np.ndarray) -> float:
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(pairwise_features_mse_DD)
    best_match_D = pairwise_features_mse_DD[row_ind, col_ind]

    return np.mean(best_match_D)


def mse(x_ND, y_ND):
    x_D1N = rearrange(x_ND, "n d -> d 1 n")
    y_1DN = rearrange(y_ND, "n d -> 1 d n")
    resid_DDN = (x_D1N - y_1DN) ** 2
    return np.mean(resid_DDN, axis=-1)


def me(x_ND, y_ND):
    x_D1N = rearrange(x_ND, "n d -> d 1 n")
    y_1DN = rearrange(y_ND, "n d -> 1 d n")
    resid_DDN = np.abs(x_D1N - y_1DN)
    return np.mean(resid_DDN, axis=-1)


def switch_error(x_ND, y_ND):
    X_D1N = rearrange(x_ND, "n d -> d 1 n")
    Y_1DN = rearrange(y_ND, "n d -> 1 d n")
    switch_error_DDN = (X_D1N > 0.5) != (Y_1DN > 0.5)
    return np.mean(switch_error_DDN, axis=-1)

def pairwise_soft_jaccard(x_ND: np.ndarray, y_ND: np.ndarray) -> float:
    x_ND1 = rearrange(x_ND, "n d -> n d 1")
    y_N1D = rearrange(y_ND, "n d -> n 1 d")
    intersection_DxDy = np.minimum(x_ND1, y_N1D).sum(axis=0)
    union_DxDy = np.maximum(x_ND1, y_N1D).sum(axis=0)
    return intersection_DxDy / (union_DxDy + 1e-10)

def softmax(x_DD):
    # Shift x_DD for numerical stability before computing softmax
    x_DD_max = np.max(x_DD, axis=1, keepdims=True)  # Shape: (N, 1)
    x_DD_shifted = x_DD - x_DD_max
    
    exp_DD = np.exp(x_DD_shifted)
    softmax_DD = exp_DD / np.sum(exp_DD, axis=1, keepdims=True)

    if not np.allclose(np.sum(softmax_DD, axis=1), 1.0, rtol=1e-5):
        raise ValueError("Input must sum to 1")
    
    return softmax_DD

def entropy(softmax_DD):
    """maps one-hot to 0, and uniform to 1"""
    return -np.sum(softmax_DD * np.log(softmax_DD), axis=1)

def vec_negentropy(p_ND: np.ndarray, epsilon: float = 1e-15) -> float:
    # Ensure input is a probability distribution
    if not np.allclose(np.sum(p_ND, axis=-1), 1.0, rtol=1e-5):
        raise ValueError("Input must sum to 1")
    
    d = p_ND.shape[1]
    # Create uniform distribution
    uniform = np.ones(d) / d
    
    # Compute entropy for input distribution
    # Add epsilon to avoid log(0)
    p_safe_ND = p_ND + epsilon
    H_p_N = -np.sum(p_safe_ND * np.log(p_safe_ND), axis=1)
    
    # Compute entropy for uniform distribution
    H_uniform = -np.sum(uniform * np.log(uniform))
    
    # Compute normalized negentropy
    J_N = (H_uniform - H_p_N) / H_uniform
    
    return J_N

# mse_feature_overlap = partial(permutation_matching, error_metric=mse)
# basic_feature_overlap = partial(permutation_matching, error_metric=me)
# switch_feature_overlap = partial(permutation_matching, error_metric=switch_error)

if __name__ == "__main__":
    # for error_metric in [mse]: # , me, switch_error]:
    #     print(f"{error_metric.__name__}:")
    #     for examples, name in [(perfect(), "perfect"), (terrible(), "terrible")]:  #, (random(16_000), "random")]:
    #         y_ND, x_ND, expected_score = examples
    #         assert x_ND.shape == y_ND.shape
    #         print("computing pairwise feature errors")
    #         pairwise_features_mse_DD = error_metric(x_ND, y_ND)
    #         permutation_score = permutation_matching(pairwise_features_mse_DD)
    #         closeness_score = closeness(pairwise_features_mse_DD)
    #         softmax_score = inverse_softmax(pairwise_features_mse_DD)
    #         print(f"{name:10} \n"
    #             # f"  permutation_score: {permutation_score:.2f}, ({expected_score:.2f})\n"
    #             # f"  closeness_score: {closeness_score:.2f}, ({expected_score:.2f})\n"
    #             f"  softmax_score: {softmax_score:.2f}")
    
    for examples, name in [(perfect(), "perfect"), (random_sparse(), "random")]:
        y_ND, x_ND = examples

        # pairwise_features_mse_DD_1 = mse(x_ND, y_ND)
        # softmax_score_1 = np.mean(softmax(1 - pairwise_features_mse_DD_1))

        # pairwise_features_mse_DD_2 = mse(y_ND, x_ND)
        # softmax_score_2 = np.mean(softmax(1 - pairwise_features_mse_DD_2))

        # print(f"{name:10} softmax_score_1: {softmax_score_1:.2f}, softmax_score_2: {softmax_score_2:.2f}")
        # print(f"  mean(softmax_score): {np.mean([softmax_score_1, softmax_score_2]):.2f}")
        print(name)
        pairwise_jaccard_similarity_DD = pairwise_soft_jaccard(x_ND, y_ND)
        acc_sm_DD = softmax(pairwise_jaccard_similarity_DD * 50)
        negent_N = vec_negentropy(acc_sm_DD)
        mean_negent_D = np.mean(negent_N)
        print(f"mean_negentropy: {mean_negent_D:.2f}")

        # softmax_score_1
        # pairwise_features_mse_DD_2 = mse(y_ND, x_ND)
        # softmax_score_2 = softmax(1 - pairwise_features_mse_DD_2)

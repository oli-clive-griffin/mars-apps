from einops import rearrange
import numpy as np

np.set_printoptions(precision=2, suppress=True)

SPARSITY = 0.1

def exact_same_features(N: int = 50, D: int = 10):
    x_ND = (np.random.random((N, D)) > SPARSITY).astype(int)
    # permute the features randomly
    # a good metric must be order invariant
    perm = np.random.permutation(D)
    y_ND = x_ND[:, perm]
    return x_ND, y_ND

def random_sparse(N: int = 50, D: int = 10):
    x_ND = (np.random.random((N, D)) > SPARSITY).astype(int)
    y_ND = (np.random.random((N, D)) > SPARSITY).astype(int)
    return x_ND, y_ND

def pairwise_soft_jaccard(x_ND: np.ndarray, y_ND: np.ndarray) -> float:
    x_ND1 = rearrange(x_ND, "n d -> n d 1")
    y_N1D = rearrange(y_ND, "n d -> n 1 d")
    intersection_DxDy = np.minimum(x_ND1, y_N1D).sum(axis=0)
    union_DxDy = np.maximum(x_ND1, y_N1D).sum(axis=0)
    return intersection_DxDy / (union_DxDy + 1e-10)

def softmax(x, axis: int = -1):
    # Shift x for numerical stability before computing softmax
    x_max = np.max(x, axis=axis, keepdims=True)  # Shape: (N, 1)
    x_shifted = x - x_max
    
    exp_x = np.exp(x_shifted)
    softmax_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    # sanity check
    if not np.allclose(np.sum(softmax_x, axis=axis), 1.0, rtol=1e-5):
        raise ValueError("Input must sum to 1")
    
    return softmax_x

def entropy(p: np.ndarray, axis: int = -1, epsilon: float = 1e-15) -> float:
    # sanity check
    if not np.allclose(np.sum(p, axis=axis), 1.0, rtol=1e-5):
        raise ValueError("Input must sum to 1")
    
    p_safe_ND = p + epsilon
    return -np.sum(p_safe_ND * np.log(p_safe_ND), axis=axis)

def normalized_negentropy(p: np.ndarray, axis: int = -1, epsilon: float = 1e-15) -> float:
    # sanity check
    if not np.allclose(np.sum(p, axis=axis), 1.0, rtol=1e-5):
        raise ValueError("Input must sum to 1")
    
    d = p.shape[axis]
    # Create uniform distribution
    uniform = np.ones(d) / d
    
    # Compute entropy for input distribution
    # Add epsilon to avoid log(0)
    p_safe_ND = p + epsilon
    H_p_N = -np.sum(p_safe_ND * np.log(p_safe_ND), axis=axis)
    
    # Compute entropy for uniform distribution
    H_uniform = -np.sum(uniform * np.log(uniform))
    
    # Compute normalized negentropy
    J_N = (H_uniform - H_p_N) / H_uniform
    
    return J_N

if __name__ == "__main__":
    for examples, name in [(exact_same_features(), "perfect"), (random_sparse(), "random")]:
        y_ND, x_ND = examples
        print(name)
        pairwise_jaccard_similarity_DD = pairwise_soft_jaccard(x_ND, y_ND)
        acc_sm_DD = softmax(pairwise_jaccard_similarity_DD * 500)

        # norm_negent_N = normalized_negentropy(acc_sm_DD)
        # mean_norm_negent_D = np.mean(norm_negent_N)
        # print(f"mean_norm_negentropy: {mean_norm_negent_D:.2f} (1 is perfect)")

        entropy_N = entropy(acc_sm_DD)
        mean_entropy_D = np.mean(entropy_N)
        print(f"mean_entropy: {mean_entropy_D:.2f} (lower is better)")

        # softmax_score_1
        # pairwise_features_mse_DD_2 = mse(y_ND, x_ND)
        # softmax_score_2 = softmax(1 - pairwise_features_mse_DD_2)

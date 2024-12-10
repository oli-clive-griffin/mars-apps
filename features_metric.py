from einops import rearrange
import numpy as np
from scipy.special import logsumexp

np.set_printoptions(precision=2, suppress=True)

SPARSITY = 0.1

# synthetic activation generation

def exact_same_features(N: int = 50, D: int = 10):
    x_ND = (np.random.random((N, D)) > SPARSITY).astype(int)
    # permute the features randomly
    # a good metric must be order invariant
    perm = np.random.permutation(D)
    y_ND = x_ND[:, perm]
    return x_ND, y_ND

def one_copied_feature(N: int = 50, D: int = 10):
    x_ND = (np.random.random((N, D)) > SPARSITY).astype(int)

    # copy one feature from the previous feature into all features in y
    idx = np.random.randint(D)
    y_ND = np.repeat(x_ND[:, idx:idx+1], D, axis=1)
    assert y_ND.shape == x_ND.shape, f"{y_ND.shape} != {x_ND.shape}"

    return x_ND, y_ND

def random_sparse(N: int = 50, D: int = 10):
    x_ND = (np.random.random((N, D)) > SPARSITY).astype(int)
    y_ND = (np.random.random((N, D)) > SPARSITY).astype(int)
    return x_ND, y_ND


# utils

def pairwise_soft_jaccard(x_ND: np.ndarray, y_ND: np.ndarray) -> float:
    x_ND1 = rearrange(x_ND, "n d -> n d 1")
    y_N1D = rearrange(y_ND, "n d -> n 1 d")
    intersection_DxDy = np.minimum(x_ND1, y_N1D).sum(axis=0)
    union_DxDy = np.maximum(x_ND1, y_N1D).sum(axis=0)
    return intersection_DxDy / (union_DxDy + 1e-10)

def softmax(x, axis: int = -1):
    """Compute softmax values over the specified axis."""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def entropy(p: np.ndarray, axis: int = -1, epsilon: float = 1e-15) -> float:
    # sanity check
    if not np.allclose(np.sum(p, axis=axis), 1.0, rtol=1e-5):
        raise ValueError("Input must sum to 1")
    
    p_safe_ND = p + epsilon
    return -np.sum(p_safe_ND * np.log(p_safe_ND), axis=axis)

def directional_thing(y_ND: np.ndarray, x_ND: np.ndarray):
    pairwise_jaccard_similarity_DD = pairwise_soft_jaccard(x_ND, y_ND)
    acc_sm_DD = softmax(pairwise_jaccard_similarity_DD * 50)

    entropy_N = entropy(acc_sm_DD)
    mean_entropy = np.mean(entropy_N)
    return mean_entropy

if __name__ == "__main__":
    for examples, name in [
        (exact_same_features(), "exact same features, shuffled across feature dimension"),
        (random_sparse(), "independent random sparse vector"),
        (one_copied_feature(), "one feature copied from the previous feature"),
    ]:
        y_ND, x_ND = examples
        print(name)
        xtoy = directional_thing(y_ND, x_ND)
        ytox = directional_thing(x_ND, y_ND)
        print(f"{xtoy:.2f}, {ytox:.2f}")

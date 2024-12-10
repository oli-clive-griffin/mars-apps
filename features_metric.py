from einops import rearrange
import numpy as np
from scipy.special import logsumexp

np.set_printoptions(precision=4, suppress=True)

SPARSITY = 0.1

# synthetic activation generation

def exact_same_features(N: int = 50, D: int = 10, sparsity: float = SPARSITY):
    x_ND = (np.random.random((N, D)) > sparsity).astype(int)
    # permute the features randomly
    # a good metric must be order invariant
    perm = np.random.permutation(D)
    y_ND = x_ND[:, perm]
    return x_ND, y_ND

def no_overlap(N: int = 50, D: int = 10, sparsity: float = SPARSITY):
    assert N % 2 == 0, "N must be even"
    x_ND = np.zeros((N, D))
    halfway = N // 2
    x_ND[halfway:, :] = 1
    y_ND = 1 - x_ND
    return x_ND, y_ND

def one_copied_feature(N: int = 50, D: int = 10, sparsity: float = SPARSITY):
    x_ND = (np.random.random((N, D)) > sparsity).astype(int)

    # copy one feature from the previous feature into all features in y
    idx = np.random.randint(D)
    y_ND = np.repeat(x_ND[:, idx:idx+1], D, axis=1)
    assert y_ND.shape == x_ND.shape, f"{y_ND.shape} != {x_ND.shape}"

    return x_ND, y_ND

def random_sparse(N: int = 50, D: int = 10, sparsity: float = SPARSITY):
    x_ND = (np.random.random((N, D)) > sparsity).astype(int)
    y_ND = (np.random.random((N, D)) > sparsity).astype(int)
    return x_ND, y_ND


def pairwise_soft_jaccard(x_ND: np.ndarray, y_ND: np.ndarray) -> np.ndarray:
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

def mean_jaccard_entropy(x_ND: np.ndarray, y_ND: np.ndarray):
    # for each pair of features jaccard similarity of activations across examples
    pairwise_jaccard_similarity_DD = pairwise_soft_jaccard(x_ND, y_ND)

    # softmax over one feature dimension. This means something like:
    # For each feature in y, 
    acc_sm_DD = softmax(pairwise_jaccard_similarity_DD * 50)
    entropy_N = entropy(acc_sm_DD)

    return np.mean(entropy_N)

if __name__ == "__main__":
    # kwargs = {
    #     "N": 10,
    #     "D": 16_768,
    #     "sparsity": 100 / 16_768,
    # }
    # for examples, name in [
    #     (exact_same_features(**kwargs), "exact same features, shuffled across feature dimension"), # type: ignore
    #     (random_sparse(**kwargs), "independent random sparse vector"), # type: ignore
    #     (one_copied_feature(**kwargs), "one feature copied from the previous feature"), # type: ignore
    #     (no_overlap(**kwargs), "no overlap"), # type: ignore
    # ]:
    #     y_ND, x_ND = examples
    #     print(name)

    #     # xtoy = directional_thing(y_ND, x_ND)
    #     # ytox = directional_thing(x_ND, y_ND)
    #     # print(f"x -> y: {xtoy:.4f}, y -> x: {ytox:.4f}")
    #     # note: this is producing 2.3026 for no the no overlap case (this is the entropy of a uniform distribution over 10 features, or log(10))
    #     # print(entropy(np.expand_dims(np.ones((10,)) / 10, axis=0)))

    #     # xtoy = directional_thing_mean(y_ND, x_ND)
    #     # ytox = directional_thing_mean(x_ND, y_ND)
    #     # print(f"x -> y: {xtoy:.4f}, y -> x: {ytox:.4f}")

    #     xtoy = mean_jaccard_entropy(y_ND, x_ND)
    #     ytox = mean_jaccard_entropy(x_ND, y_ND)
    #     print(f"x -> y: {xtoy:.4f}, y -> x: {ytox:.4f}")

    #     print()


    print(entropy(softmax(np.expand_dims(np.ones((16_768,)), axis=0))))

    base = np.zeros((16_768,))
    base[:100] = 1
    print(entropy(softmax(np.expand_dims(base, axis=0))))

    base = np.zeros((16_768,))
    base[0] = 1
    print(entropy(softmax(np.expand_dims(base, axis=0))))

    asdf = softmax(np.expand_dims(np.array([1, 0, 0, 0]), axis=0) * np.sqrt(4))
    print(asdf)
    print(entropy(asdf))

    asdf = softmax(np.expand_dims(np.array([1, 0, 0, 0, 0]), axis=0) * np.sqrt(5))
    print(asdf)
    print(entropy(asdf))

    asdf = softmax(np.expand_dims(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=0) * np.sqrt(10))
    print(asdf)
    print(entropy(asdf))


"""CS 577 HW3 student implementation. All required work uses NumPy float64.

Complete the eight TODO functions. Preserve names, arguments, return order,
and required cache keys. Boolean masks in THIS assignment use True=allowed.
Do not import an attention/Transformer implementation into this module.
"""
from typing import Dict, Optional, Tuple
import numpy as np

def masked_softmax(scores: np.ndarray,
                   allowed: Optional[np.ndarray] = None) -> np.ndarray:
    """Stable softmax over the last axis; input rank >= 2, finite float64.

    allowed is None or a Boolean array broadcastable TO scores.shape.
    True permits a key; False blocks it. Blocked probabilities are exactly
    zero. Nonempty rows sum to one. A fully blocked row returns all zeros
    without NaNs or warnings (an explicit guard, not ordinary softmax).
    Reject non-Boolean/non-broadcastable masks with ValueError. No mutation.
    """
    if scores.ndim<2:
        raise ValueError("scores must have rank >= 2")
    if scores.dtype != np.float64:
        raise ValueError("scores must be of type float64")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if allowed is not None:
        if allowed.dtype != np.bool_:
            raise ValueError("allowed must be a Boolean array")
        try:
            allowed = np.broadcast_to(allowed, scores.shape)
        except ValueError:
            raise ValueError("allowed must be broadcastable to scores.shape")
    if allowed is not None:
        scores = np.where(allowed, scores, -np.inf)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_max = np.where(np.isfinite(scores_max), scores_max, 0.0)
    scores_exp = np.exp(scores - scores_max)
    if allowed is not None:
        scores_exp = np.where(allowed, scores_exp, 0.0)
    row_sums = np.sum(scores_exp, axis=-1, keepdims=True)

    # Avoid division by zero for fully blocked rows
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    softmax = scores_exp / row_sums
    return softmax


def causal_mask(length: int) -> np.ndarray:
    """Return Boolean (T,T), True at (i,j) iff j <= i; zero-based positions.

    Include the diagonal: the input at i predicts the NEXT target token.
    length must be a positive integer. No prefix offsets in this homework.
    """
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)):
        raise ValueError("length must be a positive integer")
    if length <= 0:
        raise ValueError("length must be a positive integer")
   
    return np.tril(np.ones((length, length), dtype=np.bool_))


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 allowed: Optional[np.ndarray] = None
                                 ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Q:(B,H,Tq,dk), K:(B,H,Tk,dk), V:(B,H,Tk,dv).

    Require matching batch/head axes, matching query/key feature widths,
    and matching key/value token counts. Tq and Tk may differ. dk and dv
    may differ. Return (output, cache), output:(B,H,Tq,dv), cache containing
    scores (scaled, BEFORE masking) and weights, both (B,H,Tq,Tk).
    Invalid shapes raise ValueError. Use no loops over tokens/batches/heads.
    """
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("Q, K, V must be 4-dimensional arrays")
    B, H, Tq, dk = Q.shape
    B2, H2, Tk, dk2 = K.shape
    B3, H3, Tk2, dv = V.shape
    if B != B2 or B != B3:
        raise ValueError("Batch sizes must match")
    if H != H2 or H != H3:
        raise ValueError("Number of heads must match")
    if dk != dk2:
        raise ValueError("Query and key feature widths must match")
    if dk <= 0:
        raise ValueError("dk must be positive")
    if Tk != Tk2:
        raise ValueError("Key and value token counts must match")  
    cache = {}
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(dk)
    weights = masked_softmax(scores, allowed=allowed)
    output = np.matmul(weights, V)
    cache['scores'] = scores
    cache['weights'] = weights

    return output, cache


def split_heads(X: np.ndarray, num_heads: int) -> np.ndarray:
    """(B,T,D) -> (B,H,T,D/H), after the learned projection.

    Require positive integer H dividing D; invalid requests raise ValueError.
    Keep consecutive projected feature groups together. No input mutation.
    """
    if X.ndim != 3:
        raise ValueError("X must be 3-dimensional array")
    B, T, D = X.shape
    if isinstance(num_heads, bool) or not isinstance(num_heads, (int, np.integer)):
        raise ValueError("num_heads must be a positive integer")
    if num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    if D % num_heads != 0:
        raise ValueError("D must be divisible by num_heads")
    dh = D // num_heads
    X = X.reshape(B, T, num_heads, dh)
    X = X.transpose(0, 2, 1, 3)
    return X


def merge_heads(X: np.ndarray) -> np.ndarray:
    """(B,H,T,dh) -> (B,T,H*dh); exact inverse of split_heads."""
    if X.ndim != 4:
        raise ValueError("X must be 4-dimensional array")
    B, H, T, dh = X.shape
    X = X.transpose(0, 2, 1, 3)
    X = X.reshape(B, T, H * dh)

    return X


def multi_head_attention(X_query: np.ndarray, X_context: Optional[np.ndarray],
                         params: Dict[str, np.ndarray], num_heads: int,
                         allowed: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """X_query:(B,Tq,D); context:(B,Tk,D), or None for self-attention.

    params contains Wq,Wk,Wv,Wo, each (D,D), with no projection biases.
    Apply the learned full-width projections BEFORE splitting heads.
    Every head has dh=D/H features for queries, keys, and values.
    Concatenate head outputs in head order, then apply Wo.
    Return (Y,cache), Y:(B,Tq,D). Required cache keys:
      Q,K,V: (B,H,Tq,dh), (B,H,Tk,dh), (B,H,Tk,dh)
      scores,weights: (B,H,Tq,Tk)
      head_outputs: (B,H,Tq,dh); merged: (B,Tq,D).
    Validate dimensions. Do not mutate input arrays or parameters.
    """
    if X_query.ndim != 3:
        raise ValueError("X_query must be 3-dimensional array")
    B, Tq, D = X_query.shape
    if X_context is None:
        X_context = X_query
    if X_context.ndim != 3:
        raise ValueError("X_context must be 3-dimensional array")
    B2, Tk, D2 = X_context.shape
    if B != B2:
        raise ValueError("Batch sizes must match")
    if D != D2:
        raise ValueError("Feature dimensions must match")
    if isinstance(num_heads, bool) or not isinstance(num_heads, (int, np.integer)):
        raise ValueError("num_heads must be a positive integer")
    if num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    if D % num_heads != 0:
        raise ValueError("D must be divisible by num_heads")
    for name in ('Wq', 'Wk', 'Wv', 'Wo'):
        if name not in params:
            raise ValueError(f"params must contain '{name}'")
        if params[name].shape != (D, D):
            raise ValueError(f"params['{name}'] must have shape (D, D)")
    Q = split_heads(X_query @ params['Wq'], num_heads)
    K = split_heads(X_context @ params['Wk'], num_heads)
    V = split_heads(X_context @ params['Wv'], num_heads)    
    head_outputs, attn_cache = scaled_dot_product_attention(Q, K, V, allowed)
    merged = merge_heads(head_outputs)

    Y = merged @ params['Wo']
    cache = {
        'Q': Q, 'K': K, 'V': V,
        'scores': attn_cache['scores'], 'weights': attn_cache['weights'],
        'head_outputs': head_outputs, 'merged': merged
    }

    return Y, cache


def transformer_block(X: np.ndarray, params: dict, num_heads: int,
                      allowed: Optional[np.ndarray] = None, eps: float = 1e-5
                      ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """One pre-norm block, X and result:(B,T,D), no dropout.

    N1 = layer_norm(X, ln1_gamma, ln1_beta, eps)
    U  = X + multi_head_attention(N1, None, params['mha'], H, allowed)[0]
    N2 = layer_norm(U, ln2_gamma, ln2_beta, eps)
    F  = relu(N2 @ W1 + b1) @ W2 + b2
    Z  = U + F
    Return (Z,cache). Required keys: N1,U,N2,F,Z:(B,T,D),
    weights:(B,H,T,T). Reuse your attention code and the provided layer_norm.
    W1:(D,Dff), b1:(Dff,), W2:(Dff,D), b2:(D,).
    This exercise uses ReLU and pre-norm; do not change the block order.
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 3:
        raise ValueError("X must have shape (B, T, D)")

    B, T, D = X.shape

    required_params = (
        "ln1_gamma", "ln1_beta",
        "ln2_gamma", "ln2_beta",
        "mha", "W1", "b1", "W2", "b2"
    )

    for name in required_params:
        if name not in params:
            raise ValueError(f"Missing required parameter: {name}")

    for name in ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta"):
        if params[name].shape != (D,):
            raise ValueError(f"{name} must have shape ({D},)")
  
    if params["W1"].ndim != 2 or params["W1"].shape[0] != D:
        raise ValueError("W1 must have shape (D, Dff)")

    Dff = params["W1"].shape[1]

    if params["b1"].shape != (Dff,):
        raise ValueError("b1 must have shape (Dff,)")
    if params["W2"].shape != (Dff, D):
        raise ValueError("W2 must have shape (Dff, D)")

    if params["b2"].shape != (D,):
        raise ValueError("b2 must have shape (D,)")

    N1 = layer_norm(
        X,
        params["ln1_gamma"],
        params["ln1_beta"],
        eps
    )
    U, attn_cache = multi_head_attention(
        N1, None, params["mha"], num_heads, allowed
    )

    U = X + U
    N2 = layer_norm(
        U,
        params["ln2_gamma"],
        params["ln2_beta"],
        eps
    )
    F = np.maximum(
    0, N2 @ params["W1"] + params["b1"]
    ) @ params["W2"] + params["b2"]

    Z = U + F
    cache = {
        "N1": N1,
        "U": U,
        "N2": N2,
        "F": F,
        "Z": Z,
        "weights": attn_cache["weights"]
    }

    return Z, cache
    

def sinusoidal_positions(length: int, model_dim: int) -> np.ndarray:
    """Return float64 (T,D) with zero-based positions and even D.

    For m=0,...,D/2-1: omega_m=10000**(-2*m/D).
    Column 2*m is sin(p*omega_m); 2*m+1 is cos(p*omega_m).
    length and model_dim must be positive integers; D must be even.
    """
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)):
        raise ValueError("length must be a positive integer")
    if isinstance(model_dim, (bool, np.bool_)) or not isinstance(model_dim, (int, np.integer)):
        raise ValueError("model_dim must be a positive integer")
    if length <= 0:
        raise ValueError("length must be positive")
    if model_dim <= 0 or model_dim % 2 != 0:
        raise ValueError("model_dim must be positive and even")

    positions = np.arange(length)[:, None]
    div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / (model_dim)))
    pe = np.zeros((length, model_dim))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


# PROVIDED HELPERS BELOW: use these unchanged. They are not graded TODOs.
def layer_norm(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """Normalize each token over its last-axis FEATURES, population variance."""
    X = np.asarray(X, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if X.ndim != 3 or gamma.shape != (X.shape[-1],) or beta.shape != gamma.shape:
        raise ValueError("layer_norm needs X:(B,T,D), gamma/beta:(D,)")
    if eps <= 0:
        raise ValueError("eps must be positive")
    mean = X.mean(axis=-1, keepdims=True)
    variance = ((X - mean) ** 2).mean(axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(variance + eps) * gamma + beta


def initialize_mha(model_dim: int, seed: int = 577) -> Dict[str, np.ndarray]:
    """Provide deterministic dense full-width projections, no biases."""
    if model_dim <= 0:
        raise ValueError("model_dim must be positive")
    rng = np.random.default_rng(seed)
    return {key: rng.normal(0, 1 / np.sqrt(model_dim), (model_dim, model_dim))
            for key in ("Wq", "Wk", "Wv", "Wo")}


def initialize_block(model_dim: int = 4, ff_dim: int = 8, seed: int = 577) -> dict:
    """Provide all parameters of the specified pre-norm block."""
    if ff_dim <= 0:
        raise ValueError("ff_dim must be positive")
    rng = np.random.default_rng(seed + 1)
    return {
        "mha": initialize_mha(model_dim, seed),
        "ln1_gamma": np.ones(model_dim), "ln1_beta": np.zeros(model_dim),
        "ln2_gamma": np.ones(model_dim), "ln2_beta": np.zeros(model_dim),
        "W1": rng.normal(0, 1 / np.sqrt(model_dim), (model_dim, ff_dim)),
        "b1": np.zeros(ff_dim),
        "W2": rng.normal(0, 1 / np.sqrt(ff_dim), (ff_dim, model_dim)),
        "b2": np.zeros(model_dim),
    }


def hand_case() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The / cat / sleeps; return Q,K,V with (B,H,T,d)=(1,1,3,2)."""
    Q = np.array([[1., 0.], [0., 1.], [1., 1.]])
    K = Q.copy()
    V = np.array([[1., 0.], [0., 2.], [3., 1.]])
    return Q[None, None], K[None, None], V[None, None]


def token_case() -> np.ndarray:
    """Fixed feature vectors, X:(1,3,4); semantic labels are illustrative."""
    return np.array([[[1., 0., 0., 1.],
                      [0., 1., 1., 0.],
                      [1., 1., 0., 0.]]], dtype=np.float64)

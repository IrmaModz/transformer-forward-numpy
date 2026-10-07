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
    raise NotImplementedError("Implement masked_softmax")


def causal_mask(length: int) -> np.ndarray:
    """Return Boolean (T,T), True at (i,j) iff j <= i; zero-based positions.

    Include the diagonal: the input at i predicts the NEXT target token.
    length must be a positive integer. No prefix offsets in this homework.
    """
    raise NotImplementedError("Implement causal_mask")


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
    raise NotImplementedError("Implement scaled_dot_product_attention")


def split_heads(X: np.ndarray, num_heads: int) -> np.ndarray:
    """(B,T,D) -> (B,H,T,D/H), after the learned projection.

    Require positive integer H dividing D; invalid requests raise ValueError.
    Keep consecutive projected feature groups together. No input mutation.
    """
    raise NotImplementedError("Implement split_heads")


def merge_heads(X: np.ndarray) -> np.ndarray:
    """(B,H,T,dh) -> (B,T,H*dh); exact inverse of split_heads."""
    raise NotImplementedError("Implement merge_heads")


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
    raise NotImplementedError("Implement multi_head_attention")


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
    raise NotImplementedError("Implement transformer_block")


def sinusoidal_positions(length: int, model_dim: int) -> np.ndarray:
    """Return float64 (T,D) with zero-based positions and even D.

    For m=0,...,D/2-1: omega_m=10000**(-2*m/D).
    Column 2*m is sin(p*omega_m); 2*m+1 is cos(p*omega_m).
    length and model_dim must be positive integers; D must be even.
    """
    raise NotImplementedError("Implement sinusoidal_positions")


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

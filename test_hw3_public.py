"""Representative HW3 tests, not the complete grading rubric.

Run: python -m unittest -v test_hw3_public.py
Masks here use True=allowed. Additional grading inputs use the SAME contract.
The grader also reads your calculations, reference checks, plots, and analysis.
"""
import unittest
import numpy as np
import hw3_attention as h

ATOL = 1e-10
RTOL = 1e-9


class AttentionTests(unittest.TestCase):
    def test_01_stable_softmax_and_mask(self):
        s = np.array([[1000., 999., -1000.], [4., 5., 6.]])
        before = s.copy()
        mask = np.array([[True, True, False], [False, False, False]])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            a = h.masked_softmax(s, mask)
        t = np.exp(-1.)
        np.testing.assert_allclose(a[0], [1/(1+t), t/(1+t), 0], atol=ATOL, rtol=RTOL)
        np.testing.assert_array_equal(a[1], np.zeros(3))
        np.testing.assert_array_equal(s, before)
        with self.assertRaises(ValueError):
            h.masked_softmax(s, np.ones((2, 3)))
        with self.assertRaises(ValueError):
            h.masked_softmax(s, np.ones((2, 2), dtype=bool))

    def test_02_softmax_last_axis_and_key_mask_broadcast(self):
        s = np.zeros((2, 3, 4, 5), dtype=np.float64)
        valid = np.array([[True, True, False, False, False],
                          [True, True, True, True, True]])[:, None, None, :]
        a = h.masked_softmax(s, valid)
        self.assertEqual(a.shape, s.shape)
        np.testing.assert_allclose(a[0, :, :, :2], 0.5, atol=ATOL, rtol=RTOL)
        np.testing.assert_array_equal(a[0, :, :, 2:], 0.)
        np.testing.assert_allclose(a.sum(axis=-1), 1., atol=ATOL, rtol=RTOL)

    def test_03_causal_mask_includes_self(self):
        np.testing.assert_array_equal(h.causal_mask(3),
            [[True, False, False], [True, True, False], [True, True, True]])
        self.assertEqual(h.causal_mask(3).dtype, np.bool_)
        with self.assertRaises(ValueError):
            h.causal_mask(0)

    def test_04_attention_uniform_reads_and_rectangular_shapes(self):
        Q = np.zeros((2, 2, 3, 4))
        K = np.zeros((2, 2, 5, 4))
        V = np.arange(2*2*5*6, dtype=np.float64).reshape(2, 2, 5, 6)
        out, cache = h.scaled_dot_product_attention(Q, K, V)
        self.assertEqual(out.shape, (2, 2, 3, 6))
        self.assertEqual(cache["weights"].shape, (2, 2, 3, 5))
        np.testing.assert_allclose(out, np.broadcast_to(V.mean(axis=2, keepdims=True), out.shape))
        np.testing.assert_allclose(cache["weights"], 0.2)
        with self.assertRaises(ValueError):
            h.scaled_dot_product_attention(Q, K[..., :3], V)

    def test_05_hand_case_causal_visibility(self):
        Q, K, V = h.hand_case()
        out, cache = h.scaled_dot_product_attention(Q, K, V, h.causal_mask(3))
        self.assertEqual(out.shape, (1, 1, 3, 2))
        np.testing.assert_allclose(out[0, 0, 0], V[0, 0, 0])
        np.testing.assert_array_equal(cache["weights"][0, 0][np.triu_indices(3, 1)], 0.)
        np.testing.assert_allclose(cache["weights"].sum(axis=-1), 1.)


class HeadAndBlockTests(unittest.TestCase):
    def test_06_head_axes_and_exact_roundtrip(self):
        X = np.arange(2*3*8, dtype=np.float64).reshape(2, 3, 8)
        split = h.split_heads(X, 2)
        self.assertEqual(split.shape, (2, 2, 3, 4))
        np.testing.assert_array_equal(split[:, 0], X[..., :4])
        np.testing.assert_array_equal(split[:, 1], X[..., 4:])
        np.testing.assert_array_equal(h.merge_heads(split), X)
        with self.assertRaises(ValueError):
            h.split_heads(X, 3)

    def test_07_mha_cache_and_identity_uniform_case(self):
        X = h.token_case()
        params = {"Wq": np.zeros((4, 4)), "Wk": np.zeros((4, 4)),
                  "Wv": np.eye(4), "Wo": np.eye(4)}
        before = X.copy()
        Y, cache = h.multi_head_attention(X, None, params, 2)
        np.testing.assert_allclose(Y, np.broadcast_to(X.mean(axis=1, keepdims=True), X.shape))
        for name in ("Q", "K", "V", "head_outputs"):
            self.assertEqual(cache[name].shape, (1, 2, 3, 2))
        self.assertEqual(cache["scores"].shape, (1, 2, 3, 3))
        self.assertEqual(cache["merged"].shape, X.shape)
        np.testing.assert_array_equal(X, before)

    def test_08_mha_cross_attention_and_mask(self):
        query = np.ones((2, 2, 4))
        context = np.arange(2*3*4, dtype=np.float64).reshape(2, 3, 4)
        p = {"Wq": np.zeros((4, 4)), "Wk": np.zeros((4, 4)),
             "Wv": np.eye(4), "Wo": np.eye(4)}
        allowed = np.array([[True, False, True], [True, True, False]])[:, None, None, :]
        Y, c = h.multi_head_attention(query, context, p, 2, allowed)
        self.assertEqual(Y.shape, (2, 2, 4))
        self.assertEqual(c["weights"].shape, (2, 2, 2, 3))
        np.testing.assert_allclose(Y[0], np.broadcast_to(context[0, [0, 2]].mean(0), (2, 4)))
        np.testing.assert_allclose(Y[1], np.broadcast_to(context[1, :2].mean(0), (2, 4)))

    def test_09_position_vectors(self):
        P = h.sinusoidal_positions(3, 4)
        self.assertEqual(P.shape, (3, 4))
        self.assertEqual(P.dtype, np.float64)
        np.testing.assert_allclose(P[0], [0., 1., 0., 1.], atol=ATOL)
        np.testing.assert_allclose(P[1], [np.sin(1.), np.cos(1.), np.sin(.01), np.cos(.01)], atol=ATOL)
        with self.assertRaises(ValueError):
            h.sinusoidal_positions(3, 5)

    def test_10_block_residual_identity_when_updates_zero(self):
        X = h.token_case()
        p = h.initialize_block()
        p["mha"]["Wo"][:] = 0
        p["W2"][:] = 0
        p["b2"][:] = 0
        Z, c = h.transformer_block(X, p, 2)
        np.testing.assert_allclose(Z, X, atol=ATOL, rtol=RTOL)
        for name in ("N1", "U", "N2", "F", "Z"):
            self.assertEqual(c[name].shape, X.shape)
        self.assertEqual(c["weights"].shape, (1, 2, 3, 3))

    def test_11_causal_block_future_change_cannot_change_prefix(self):
        X = h.token_case()
        altered = X.copy()
        altered[:, 2] += np.array([5., -3., 2., -4.])
        p = h.initialize_block()
        a, _ = h.transformer_block(X, p, 2, h.causal_mask(3))
        b, _ = h.transformer_block(altered, p, 2, h.causal_mask(3))
        np.testing.assert_allclose(a[:, :2], b[:, :2], atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    print("CS 577 HW3 public tests; passing does not guarantee full credit.\n")
    unittest.main(verbosity=2)

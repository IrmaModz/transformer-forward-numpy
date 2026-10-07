# CS 577 Homework 3 Rubric

| Part | Points | Full-credit evidence |
|---|---:|---|
| 0. Setup and axes | 5 | Actual versions and tests (2); axis meanings, score/output shapes, key-mask broadcast (3) |
| 1. Hand attention | 20 | Full attention values and cat-row expansion (8); causal calculation and visibility (8); scaling and stability reasoning (4) |
| 2. Stable masked attention | 25 | Masked softmax and empty-row guard (10); vectorized, rectangular attention (10); padding/causal masks and perturbation check (5) |
| 3. Multi-head attention | 25 | Full projections, split/merge, cache, maps and shape trace (12); independent PyTorch output/weight checks in three cases (6); fixed-width head-count and head-output ablations with interpretation (7) |
| 4. Transformer block | 20 | Specified pre-norm block, shape trace, independent reference and residual check (8); position vectors and permutation experiment (6); future-input perturbation and causal interpretation (6) |
| 5. Architecture and quality | 5 | Correct generation-step comparison, BOS, Q/K/V origins and prefill visibility (3); runnable, readable, correctly named files and disclosure (2) |
| **Total** | **100** | |

Core code is NumPy float64, CPU, vectorized over batches/heads/tokens. PyTorch is an independent notebook reference, not a replacement for the core code. `layer_norm` and parameter/case helpers are provided; students implement eight functions.

Public tests are representative. Additional tests follow the same interfaces and vary sizes, parameters, masks and rectangular query/context counts. No training, attention backward, KV cache, FlashAttention or GPU implementation is required.

Numerical checks normally use `atol=1e-10, rtol=1e-9`. The supplied PyTorch and causal/permutation equality checks should have maximum absolute errors below `1e-9`. Hand answers rounded to four decimals are accepted with appropriate rounding tolerance. An observed nonzero difference under a deliberate change is interpreted alongside the explanation, not used as a semantic-quality score.

Mask contract: Boolean True is allowed; blocked weights are exact zero; fully blocked rows return zeros as an explicit safe extension. Normal softmax on an entirely blocked row is undefined. Padding a query does not force a residual block's complete output to zero.

No credit for affected core functions replaced by forbidden attention/Transformer APIs. Up to half credit for affected functions that violate required vectorization. Requested reasoning missing: at most half credit for that item. Deduct missing artifacts, execution evidence or outputs within affected components; do not double-penalize a single underlying issue. Missing disclosure loses up to two quality points. Apply the syllabus academic-honesty policy to integrity violations.

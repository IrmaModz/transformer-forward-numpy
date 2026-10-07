# CS 577 Homework 3

**Attention, Multi-Head Attention, and a Transformer Forward Pass**

Release: Wednesday, September 30, 2026.  
Due: Sunday, November 8, 2026, 11:59 p.m. Central Time.  
Individual work; 100 points; 8% of the course grade; expected effort 8–12 hours.  
The same core assignment applies to in-person and asynchronous online students.

## Start here

1. Read `CS577_HW3_Assignment_and_Rubric.pdf` for the complete tasks, scoring,
   allowed assistance, and submission requirements.
2. Keep all extracted files in one folder. Install the requirements in your
   Python environment: `python -m pip install -r requirements.txt`.
3. Complete the eight TODO functions in `hw3_attention.py` and the TODO cells
   in `hw3_starter.ipynb`. Keep the provided helper functions unchanged.
4. Run `python -m unittest -v test_hw3_public.py` from that folder. The initial
   starter deliberately raises `NotImplementedError`; it is not a completed
   implementation. Public tests are representative, not a complete grading key.
5. Restart the notebook kernel and run all cells on CPU. Keep the actual
   outputs, numerical comparisons, labeled attention maps, and explanations.
6. Complete `hw3_written_template.tex` and compile your written PDF. The
   included `latex_symbols.pdf` is an optional notation reference.

Use Python 3.9 or newer. No GPU, network dataset, model training, or manual
attention backward pass is required. Core operations use NumPy float64;
PyTorch supplies an independent forward reference, not the core implementation.

## Important conventions

- In this assignment, Boolean `True` means a key is **allowed**. Softmax
  normalizes over key positions, independently for every query, head, and batch.
- Include the causal diagonal: an input position predicts the next token.
- A fully blocked row returns zeros without NaNs or warnings, as explicitly
  required by this assignment. This is a guarded extension, not ordinary softmax.
- Apply learned projections before splitting heads. Preserve the documented
  function signatures, return order, cache keys, and dimensions.
- Follow the PDF's assistance policy and include the required disclosure.
  Do not submit generated core implementations, derivations, or analysis.

## Submit

Submit one `hw3_<hawk_username>.zip` containing exactly:

- `hw3_<hawk_username>.pdf`;
- `hw3_<hawk_username>.ipynb`, with visible outputs and plots;
- `hw3_attention.py`.

Do not include environments, solution files, tests, or the original release ZIP
in your submission. Embed the attention maps in your report; a separate image
file is not required. Consult the PDF and `hw3_rubric.md` for grading details.

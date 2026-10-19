"""Guard for the numba JIT under the umap-learn / scikit-learn dependency stack.

The numba/llvmlite install matrix is the most fragile part of these dependencies (it is
sensitive to the numpy pin). A broken resolution fails the real UMAP fits and DBSCAN runs in
test_projection.py; this file catches the failure those fits cannot see: an install that
silently lost compilation.
"""

import numba


def test_numba_jit_is_active() -> None:
    """A working import is not a working JIT.

    UMAP produces correct output with the JIT disabled — numba just runs the kernels as plain
    Python — so the fits in test_projection.py cannot tell a healthy install from one that
    silently lost compilation and got orders of magnitude slower. Assert compilation directly.
    """
    assert numba.config.DISABLE_JIT == 0, "numba JIT is disabled; UMAP would fall back to Python speed"

    @numba.njit(cache=False)
    def _add(a: int, b: int) -> int:
        return a + b

    assert _add(1, 2) == 3
    # Non-empty only if the call actually compiled a specialization.
    assert _add.signatures, "numba did not compile the test kernel"

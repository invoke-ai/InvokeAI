"""Smoke test for the umap-learn / scikit-learn / numba dependency stack.

The numba/llvmlite install matrix is the most fragile part of these dependencies (it is
sensitive to the numpy pin), so this test exists to fail loudly in CI if resolution or the JIT
is broken, before any feature code relies on them.

Not marked `slow`: measured at ~10s cold (numba JIT cache empty) and ~7s warm, which the
default selection can absorb.
"""

import numba
import numpy as np


def test_numba_jit_is_active() -> None:
    """A working import is not a working JIT.

    UMAP produces correct output with the JIT disabled — numba just runs the kernels as plain
    Python — so the fit below cannot tell a healthy install from one that silently lost
    compilation and got orders of magnitude slower. Assert compilation directly.
    """
    assert numba.config.DISABLE_JIT == 0, "numba JIT is disabled; UMAP would fall back to Python speed"

    @numba.njit(cache=False)
    def _add(a: int, b: int) -> int:
        return a + b

    assert _add(1, 2) == 3
    # Non-empty only if the call actually compiled a specialization.
    assert _add.signatures, "numba did not compile the test kernel"


def test_umap_and_dbscan_smoke() -> None:
    from sklearn.cluster import DBSCAN
    from umap import UMAP

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((50, 16)).astype(np.float32)

    coords = UMAP(n_neighbors=10, n_components=2, min_dist=0.05, metric="cosine").fit_transform(embeddings)
    coords = np.asarray(coords)
    assert coords.shape == (50, 2)
    assert np.isfinite(coords).all()

    labels = DBSCAN(eps=0.5, min_samples=5).fit(coords).labels_
    assert labels.shape == (50,)

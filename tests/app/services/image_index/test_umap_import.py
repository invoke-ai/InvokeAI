"""Smoke test for the umap-learn / scikit-learn / numba dependency stack.

The numba/llvmlite install matrix is the most fragile part of these
dependencies (it is sensitive to the numpy pin), so this test exists to fail
loudly in CI if resolution or the JIT is broken, before any feature code
relies on them. The first UMAP fit also JIT-compiles numba kernels, which is
the slow part — hence the `slow` marker consideration; it typically takes
tens of seconds on a cold cache.
"""

import numpy as np


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

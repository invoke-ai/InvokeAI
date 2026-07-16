import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import { describe, expect, it } from 'vitest';

import { createDerivedSurfaceCache } from './derivedSurfaceCache';
import { createLayerCacheStore } from './layerCache';
import { createTestStubRasterBackend } from './raster.testStub';
import { enforceSurfaceBudget } from './surfaceBudget';

describe('enforceSurfaceBudget', () => {
  it('evicts derived surfaces before hidden base surfaces', () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const derived = createDerivedSurfaceCache();
    layers.getOrCreate('visible', 10, 10);
    layers.getOrCreate('hidden', 10, 10);
    const source = backend.createSurface(10, 10);
    derived.get({
      create: () => backend.createSurface(10, 10),
      kind: 'mask-fill',
      layerId: 'visible',
      paramsKey: 'red',
      source,
      sourceVersion: 0,
    });

    const result = enforceSurfaceBudget(layers, derived, ['visible'], 800);

    expect(result.evictedDerivedLayerIds).toEqual(['visible']);
    expect(result.evictedBaseLayerIds).toEqual([]);
    expect(layers.get('hidden')).toBeDefined();
  });

  it('protects visible base pixels and reports their overage', () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const derived = createDerivedSurfaceCache();
    const diagnostics = createCanvasDiagnostics(true);
    layers.getOrCreate('visible', 20, 20);

    const result = enforceSurfaceBudget(layers, derived, ['visible'], 400, diagnostics);

    expect(result.overBudgetVisibleBaseBytes).toBe(1_200);
    expect(layers.get('visible')).toBeDefined();
    expect(diagnostics.snapshot().overBudgetVisibleBaseBytes).toBe(1_200);
  });
});

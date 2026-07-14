import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import { describe, expect, it } from 'vitest';

import type { StubRasterSurface } from './raster.testStub';

import { createDerivedSurfaceCache } from './derivedSurfaceCache';
import { createTestStubRasterBackend } from './raster.testStub';

describe('createDerivedSurfaceCache', () => {
  it('reuses an unchanged effect and rebuilds exactly once for a changed key', () => {
    const backend = createTestStubRasterBackend();
    const source = backend.createSurface(10, 20);
    const cache = createDerivedSurfaceCache();
    let builds = 0;
    const get = (sourceVersion: number, paramsKey: string) =>
      cache.get({
        create: (target) => {
          builds += 1;
          return target ?? backend.createSurface(source.width, source.height);
        },
        kind: 'adjustments',
        layerId: 'a',
        paramsKey,
        source,
        sourceVersion,
      });

    const first = get(1, 'bright');
    expect(get(1, 'bright')).toBe(first);
    expect(builds).toBe(1);

    expect(get(2, 'bright')).toBe(first);
    expect(get(2, 'contrast')).toBe(first);
    expect(builds).toBe(3);
  });

  it('keeps unrelated layer effects and removes every effect for a deleted layer', () => {
    const backend = createTestStubRasterBackend();
    const cache = createDerivedSurfaceCache();
    const create = (layerId: string, kind: 'mask-fill' | 'control-transparency') => {
      const source = backend.createSurface(4, 4);
      return cache.get({
        create: () => backend.createSurface(4, 4),
        kind,
        layerId,
        paramsKey: 'x',
        source,
        sourceVersion: 0,
      });
    };

    create('a', 'mask-fill');
    create('a', 'control-transparency');
    const b = create('b', 'mask-fill');
    cache.deleteLayer('a');

    expect(cache.size()).toBe(1);
    expect(cache.byteSize()).toBe(4 * 4 * 4);
    expect(create('b', 'mask-fill')).not.toBe(b); // different source identity is guarded
  });

  it('evicts least-recently-used entries to a deterministic byte budget', () => {
    const backend = createTestStubRasterBackend();
    const cache = createDerivedSurfaceCache();
    const surfaces = new Map<string, StubRasterSurface>();
    const get = (layerId: string) => {
      const source = surfaces.get(layerId) ?? backend.createSurface(10, 10);
      surfaces.set(layerId, source);
      return cache.get({
        create: () => backend.createSurface(10, 10),
        kind: 'mask-fill',
        layerId,
        paramsKey: 'solid:red',
        source,
        sourceVersion: 0,
      });
    };

    get('a');
    get('b');
    get('a'); // a is now most recently used
    expect(cache.evictToBudget(400)).toEqual(['b']);
    expect(cache.byteSize()).toBe(400);
  });

  it('reports cache hits, misses, evictions, and allocated bytes when diagnostics are enabled', () => {
    const backend = createTestStubRasterBackend();
    const diagnostics = createCanvasDiagnostics(true);
    const cache = createDerivedSurfaceCache(diagnostics);
    const source = backend.createSurface(5, 5);
    const request = {
      create: () => backend.createSurface(5, 5),
      kind: 'adjustments' as const,
      layerId: 'a',
      paramsKey: 'x',
      source,
      sourceVersion: 0,
    };
    cache.get(request);
    cache.get(request);
    cache.evictToBudget(0);

    expect(diagnostics.snapshot()).toMatchObject({
      allocatedDerivedBytes: 100,
      derivedCacheEvictions: 1,
      derivedCacheHits: 1,
      derivedCacheMisses: 1,
    });
  });
});

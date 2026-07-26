import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterizationJob } from '@workbench/canvas-engine/controllers/rasterController';
import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createLayerRasterizer, type CreateLayerRasterizerDeps } from './layerRasterizer';

const surface = (width = 4, height = 4): RasterSurface =>
  ({
    canvas: {},
    ctx: { clearRect: vi.fn(), drawImage: vi.fn(), setTransform: vi.fn() },
    height,
    resize: vi.fn(),
    width,
  }) as unknown as RasterSurface;

const rasterLayer = (overrides: Record<string, unknown> = {}): CanvasLayerContract =>
  ({
    id: 'layer-1',
    isEnabled: true,
    isLocked: false,
    name: 'Layer 1',
    source: { image: { height: 4, imageName: 'a.png', width: 4 }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
    ...overrides,
  }) as unknown as CanvasLayerContract;

const textLayer = (): CanvasLayerContract =>
  rasterLayer({
    source: {
      color: '#fff',
      content: 'hello',
      fontFamily: 'Inter',
      fontSize: 12,
      fontStyle: 'normal',
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text',
    },
  });

const documentOf = (layers: CanvasLayerContract[]): CanvasDocumentContractV2 =>
  ({ height: 64, layers, width: 64 }) as unknown as CanvasDocumentContractV2;

interface Harness {
  deps: CreateLayerRasterizerDeps;
  entry: LayerCacheEntry;
  installed: Map<string, RasterizationJob>;
  documentGeneration: { value: number };
  document: { value: CanvasDocumentContractV2 };
  state: { disposed: boolean; hasCanvas: boolean };
  rasterize: ReturnType<typeof vi.fn>;
  fontReady: (() => void) | null;
  spies: {
    cancel: ReturnType<typeof vi.fn>;
    invalidateLayerCache: ReturnType<typeof vi.fn>;
    invalidateLayerRender: ReturnType<typeof vi.fn>;
    releaseBitmapIfUnreferenced: ReturnType<typeof vi.fn>;
    reportError: ReturnType<typeof vi.fn>;
    setStatus: ReturnType<typeof vi.fn>;
    setVersion: ReturnType<typeof vi.fn>;
    trackPublishedLayerImage: ReturnType<typeof vi.fn>;
  };
}

let harness: Harness;

const makeHarness = (overrides: Partial<CreateLayerRasterizerDeps> = {}): Harness => {
  const entry = {
    layerId: 'layer-1',
    rect: { height: 4, width: 4, x: 0, y: 0 },
    surface: surface(),
    version: 3,
  } as unknown as LayerCacheEntry;
  const installed = new Map<string, RasterizationJob>();
  const documentGeneration = { value: 7 };
  const document = { value: documentOf([rasterLayer()]) };
  const state = { disposed: false, hasCanvas: true };
  const rasterize = vi.fn(() => Promise.resolve({ rect: { height: 4, width: 4, x: 0, y: 0 }, surface: surface() }));
  const spies = {
    cancel: vi.fn(),
    invalidateLayerCache: vi.fn(),
    invalidateLayerRender: vi.fn(),
    releaseBitmapIfUnreferenced: vi.fn(),
    reportError: vi.fn(),
    setStatus: vi.fn(),
    setVersion: vi.fn(),
    trackPublishedLayerImage: vi.fn(),
  };
  const result: Harness = {
    document,
    documentGeneration,
    entry,
    fontReady: null,
    installed,
    rasterize,
    spies,
    state,
    deps: {
      createSurface: (width, height) => surface(width, height),
      fontLoader: {
        ensure: (_font, onReady) => {
          result.fontReady = onReady;
        },
      },
      getDocument: () => document.value,
      hasCanvasState: () => state.hasCanvas,
      invalidateLayerCache: spies.invalidateLayerCache,
      invalidateLayerRender: spies.invalidateLayerRender,
      isDisposed: () => state.disposed,
      jobs: {
        cancel: spies.cancel,
        finish: (layerId, job) => {
          if (installed.get(layerId) === job) {
            installed.delete(layerId);
          }
        },
        get: (layerId) => installed.get(layerId),
        getDocumentGeneration: () => documentGeneration.value,
        install: (layerId, job) => installed.set(layerId, job),
      },
      layerCache: {
        get: vi.fn(() => entry),
        getOrCreateRect: vi.fn(() => entry),
        publishPixels: vi.fn(() => ({ ...entry, version: entry.version + 1 })),
        version: vi.fn(() => entry.version),
      } as unknown as CreateLayerRasterizerDeps['layerCache'],
      rasterize: rasterize as unknown as CreateLayerRasterizerDeps['rasterize'],
      releaseBitmapIfUnreferenced: spies.releaseBitmapIfUnreferenced,
      reportError: spies.reportError,
      thumbnails: { setStatus: spies.setStatus, setVersion: spies.setVersion },
      trackPublishedLayerImage: spies.trackPublishedLayerImage,
      ...overrides,
    },
  };
  return result;
};

beforeEach(() => {
  harness = makeHarness();
});

const start = (layer = rasterLayer(), signal?: AbortSignal) =>
  createLayerRasterizer(harness.deps).getOrStartLayerRasterization(layer, harness.document.value, signal);

describe('refusals before any work', () => {
  it('reports aborted for an already-aborted signal', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(start(rasterLayer(), controller.signal)).resolves.toBe('aborted');
    expect(harness.rasterize).not.toHaveBeenCalled();
  });

  it.each([
    ['the engine is disposed', (h: Harness) => (h.state.disposed = true)],
    ['there is no canvas state', (h: Harness) => (h.state.hasCanvas = false)],
  ])('reports stale when %s', async (_label, mutate) => {
    mutate(harness);
    await expect(start()).resolves.toBe('stale');
    expect(harness.rasterize).not.toHaveBeenCalled();
  });

  it('reports stale for a source it cannot rasterize', async () => {
    await expect(start(rasterLayer({ source: { kind: 'polygon', points: [], type: 'shape' } }))).resolves.toBe('stale');
  });
});

describe('job reuse', () => {
  it('joins an in-flight job describing the same source, version and generation', async () => {
    const rasterizer = createLayerRasterizer(harness.deps);
    const first = rasterizer.getOrStartLayerRasterization(rasterLayer(), harness.document.value);
    const second = rasterizer.getOrStartLayerRasterization(rasterLayer(), harness.document.value);
    expect(second).toBe(first);
    expect(harness.rasterize).toHaveBeenCalledTimes(1);
    await first;
  });

  it.each([
    ['the cache version moved', (h: Harness) => ((h.entry as { version: number }).version = 99)],
    ['the document generation moved', (h: Harness) => (h.documentGeneration.value += 1)],
  ])('restarts rather than joining once %s', async (_label, mutate) => {
    const rasterizer = createLayerRasterizer(harness.deps);
    const first = rasterizer.getOrStartLayerRasterization(rasterLayer(), harness.document.value);
    mutate(harness);
    const second = rasterizer.getOrStartLayerRasterization(rasterLayer(), harness.document.value);
    expect(second).not.toBe(first);
    expect(harness.spies.cancel).toHaveBeenCalledWith('layer-1');
    await Promise.all([first, second]);
  });

  it('restarts when the source changed by value even though nothing else did', async () => {
    const rasterizer = createLayerRasterizer(harness.deps);
    const first = rasterizer.getOrStartLayerRasterization(rasterLayer(), harness.document.value);
    const edited = rasterLayer({ source: { image: { height: 4, imageName: 'b.png', width: 4 }, type: 'image' } });
    const second = rasterizer.getOrStartLayerRasterization(edited, harness.document.value);
    expect(second).not.toBe(first);
    await Promise.all([first, second]);
  });

  it('cancels the shared job when a joining caller aborts, and unhooks its listener', async () => {
    const rasterizer = createLayerRasterizer(harness.deps);
    const first = rasterizer.getOrStartLayerRasterization(rasterLayer(), harness.document.value);
    const controller = new AbortController();
    const removeSpy = vi.spyOn(controller.signal, 'removeEventListener');
    const joined = rasterizer.getOrStartLayerRasterization(rasterLayer(), harness.document.value, controller.signal);
    const job = harness.installed.get('layer-1')!;
    controller.abort();
    expect(job.abortedByCaller).toBe(true);
    await Promise.all([first, joined]);
    expect(removeSpy).toHaveBeenCalled();
  });
});

describe('publishing', () => {
  it('writes the pixels, publishes, and reports the layer ready', async () => {
    await expect(start()).resolves.toBe('published');
    expect(harness.spies.trackPublishedLayerImage).toHaveBeenCalled();
    expect(harness.spies.setVersion).toHaveBeenCalledWith('layer-1', 4);
    expect(harness.spies.setStatus).toHaveBeenCalledWith('layer-1', 'ready');
    expect(harness.spies.invalidateLayerRender).toHaveBeenCalledWith('layer-1');
  });

  it('keeps the decoded bitmap it published', async () => {
    await start();
    expect(harness.spies.releaseBitmapIfUnreferenced).not.toHaveBeenCalled();
  });

  it('resizes the cache surface when the raster came back a different size', async () => {
    harness.rasterize.mockResolvedValue({ rect: { height: 8, width: 8, x: 0, y: 0 }, surface: surface(8, 8) });
    await start();
    expect(harness.entry.surface.resize).toHaveBeenCalledWith(8, 8);
  });

  it('clears but does not draw an empty result', async () => {
    harness.rasterize.mockResolvedValue({ rect: { height: 0, width: 0, x: 0, y: 0 }, surface: surface(0, 0) });
    await expect(start()).resolves.toBe('published');
    expect(harness.entry.surface.ctx.drawImage).not.toHaveBeenCalled();
    expect(harness.entry.surface.ctx.clearRect).toHaveBeenCalled();
  });

  it('copies the result rect rather than aliasing it', async () => {
    const rect = { height: 4, width: 4, x: 1, y: 2 };
    harness.rasterize.mockResolvedValue({ rect, surface: surface() });
    await start();
    expect(harness.entry.rect).toEqual(rect);
    expect(harness.entry.rect).not.toBe(rect);
  });

  it('reports stale when the cache refuses to publish', async () => {
    (harness.deps.layerCache.publishPixels as ReturnType<typeof vi.fn>).mockReturnValue(undefined);
    await expect(start()).resolves.toBe('stale');
    expect(harness.spies.setStatus).not.toHaveBeenCalled();
  });
});

describe('a result that no longer describes the live layer', () => {
  it.each([
    ['the engine was disposed', (h: Harness) => (h.state.disposed = true)],
    ['the canvas state went away', (h: Harness) => (h.state.hasCanvas = false)],
    ['the document generation moved', (h: Harness) => (h.documentGeneration.value += 1)],
    ['the layer was deleted', (h: Harness) => (h.document.value = documentOf([]))],
    [
      'the layer source was edited',
      (h: Harness) =>
        (h.document.value = documentOf([
          rasterLayer({ source: { image: { height: 4, imageName: 'b.png', width: 4 }, type: 'image' } }),
        ])),
    ],
    [
      'the cache entry was dropped',
      (h: Harness) => (h.deps.layerCache.get as ReturnType<typeof vi.fn>).mockReturnValue(undefined),
    ],
    [
      'the cache version moved under it',
      (h: Harness) => (h.deps.layerCache.get as ReturnType<typeof vi.fn>).mockReturnValue({ ...h.entry, version: 99 }),
    ],
  ])('is dropped as stale once %s', async (_label, mutate) => {
    harness.rasterize.mockImplementation(() => {
      mutate(harness);
      return Promise.resolve({ rect: { height: 4, width: 4, x: 0, y: 0 }, surface: surface() });
    });
    await expect(start()).resolves.toBe('stale');
    expect(harness.spies.setVersion).not.toHaveBeenCalled();
  });

  it('is dropped as stale when a newer job replaced it', async () => {
    harness.rasterize.mockImplementation(() => {
      harness.installed.set('layer-1', { version: 999 } as unknown as RasterizationJob);
      return Promise.resolve({ rect: { height: 4, width: 4, x: 0, y: 0 }, surface: surface() });
    });
    await expect(start()).resolves.toBe('stale');
  });

  it('releases the bitmap it pulled in but never published', async () => {
    harness.rasterize.mockImplementation(() => {
      harness.state.disposed = true;
      return Promise.resolve({ rect: { height: 4, width: 4, x: 0, y: 0 }, surface: surface() });
    });
    await start();
    expect(harness.spies.releaseBitmapIfUnreferenced).toHaveBeenCalledWith('a.png');
  });
});

describe('failure', () => {
  it('marks the layer errored and reports it', async () => {
    harness.rasterize.mockRejectedValue(new Error('decode failed'));
    await expect(start()).resolves.toBe('error');
    expect(harness.spies.setStatus).toHaveBeenCalledWith('layer-1', 'error');
    expect(harness.spies.reportError).toHaveBeenCalledWith(
      'Layer thumbnail rasterization failed',
      'layer-1',
      expect.any(Error)
    );
  });

  it('still reports error when diagnostics itself throws', async () => {
    harness.rasterize.mockRejectedValue(new Error('decode failed'));
    harness.spies.reportError.mockImplementation(() => {
      throw new Error('diagnostics is down');
    });
    await expect(start()).resolves.toBe('error');
  });

  it('stays quiet about a failure whose layer has already moved on', async () => {
    harness.rasterize.mockImplementation(() => {
      harness.document.value = documentOf([]);
      return Promise.reject(new Error('decode failed'));
    });
    await expect(start()).resolves.toBe('stale');
    expect(harness.spies.reportError).not.toHaveBeenCalled();
    expect(harness.spies.setStatus).not.toHaveBeenCalled();
  });

  it.each([
    [
      'the cache version moved on',
      (h: Harness) => (h.deps.layerCache.version as ReturnType<typeof vi.fn>).mockReturnValue(99),
    ],
    [
      'the layer source was edited',
      (h: Harness) =>
        (h.document.value = documentOf([
          rasterLayer({ source: { image: { height: 4, imageName: 'b.png', width: 4 }, type: 'image' } }),
        ])),
    ],
  ])('stays quiet about a failure once %s', async (_label, mutate) => {
    harness.rasterize.mockImplementation(() => {
      mutate(harness);
      return Promise.reject(new Error('decode failed'));
    });
    await expect(start()).resolves.toBe('stale');
    expect(harness.spies.reportError).not.toHaveBeenCalled();
    expect(harness.spies.setStatus).not.toHaveBeenCalled();
  });

  it('reports aborted, not error, when the caller cancelled', async () => {
    const controller = new AbortController();
    harness.rasterize.mockImplementation(() => {
      controller.abort();
      return Promise.reject(new Error('aborted'));
    });
    await expect(start(rasterLayer(), controller.signal)).resolves.toBe('aborted');
    expect(harness.spies.reportError).not.toHaveBeenCalled();
  });
});

describe('text sources', () => {
  it('re-invalidates once the real font arrives', async () => {
    harness.document.value = documentOf([textLayer()]);
    await start(textLayer());
    expect(harness.fontReady).toBeTypeOf('function');
    harness.fontReady?.();
    expect(harness.spies.invalidateLayerCache).toHaveBeenCalledWith('layer-1');
    expect(harness.spies.invalidateLayerRender).toHaveBeenCalledWith('layer-1');
  });

  it('does not re-invalidate when the font arrives after the layer moved on', async () => {
    harness.document.value = documentOf([textLayer()]);
    await start(textLayer());
    harness.document.value = documentOf([]);
    harness.fontReady?.();
    expect(harness.spies.invalidateLayerCache).not.toHaveBeenCalled();
  });

  it('does not wait on fonts for a non-text source', async () => {
    await start();
    expect(harness.fontReady).toBeNull();
  });
});

import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { SamPreviewState } from '@workbench/canvas-engine/controllers/previewStateController';
import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { PreviewStateController } from '@workbench/canvas-engine/controllers/previewStateController';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import * as compositor from '@workbench/canvas-engine/render/compositor';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { FloatingSelectionFrame } from './floatingSelectionFrame';

import { createCompositeFrame, type CreateCompositeFrameDeps } from './compositeFrame';

const VIEW = [1, 0, 0, 1, 0, 0] as unknown as Mat2d;

const surface = (width = 64, height = 64): RasterSurface =>
  ({
    canvas: {},
    ctx: { clearRect: vi.fn(), drawImage: vi.fn(), setTransform: vi.fn() },
    height,
    width,
  }) as unknown as RasterSurface;

const layer = (id: string, overrides: Record<string, unknown> = {}): CanvasLayerContract =>
  ({
    id,
    isEnabled: true,
    name: id,
    source: { image: { height: 8, imageName: `${id}.png`, width: 8 }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
    ...overrides,
  }) as unknown as CanvasLayerContract;

const documentOf = (layers: CanvasLayerContract[]): CanvasDocumentContractV3 =>
  ({
    bbox: { height: 32, width: 32, x: 4, y: 6 },
    height: 64,
    stacks: stacksFrom(layers),
    width: 64,
  }) as unknown as CanvasDocumentContractV3;

interface Harness {
  deps: CreateCompositeFrameDeps;
  stores: ReturnType<typeof createEngineStores>;
  previews: PreviewStateController;
  overrides: Map<string, { x: number; y: number }>;
  entries: Map<string, LayerCacheEntry>;
  rasterizeLayer: ReturnType<typeof vi.fn>;
  beginFrame: ReturnType<typeof vi.fn>;
  enforce: ReturnType<typeof vi.fn>;
}

let harness: Harness;
let composite: ReturnType<typeof vi.spyOn>;
const FRAME_USAGE = { derivedTick: 3, groupTick: 5 };

const entryFor = (rect = { height: 8, width: 8, x: 0, y: 0 }, stale = false) =>
  ({ rect, stale, surface: surface(8, 8), version: 1 }) as unknown as LayerCacheEntry;

const makeHarness = (): Harness => {
  const stores = createEngineStores();
  const previews = new PreviewStateController();
  const overrides = new Map<string, { x: number; y: number }>();
  const entries = new Map<string, LayerCacheEntry>([['a', entryFor()]]);
  const rasterizeLayer = vi.fn();
  const beginFrame = vi.fn(() => FRAME_USAGE);
  const enforce = vi.fn(() => ({ evictedBaseLayerIds: [] as string[], overageBytes: 0 }));
  return {
    beginFrame,
    deps: {
      backend: {} as CreateCompositeFrameDeps['backend'],
      derivedSurfaceCache: { byteSize: () => 0 } as CreateCompositeFrameDeps['derivedSurfaceCache'],
      diagnostics: {} as CreateCompositeFrameDeps['diagnostics'],
      getAdjustedSurface: () => null,
      getGroupSurface: () => null,
      getCheckerboardTile: () => surface(8, 8),
      getMaskPatternTile: () => null,
      layerCache: {
        byteSize: () => 0,
        get: (id: string) => entries.get(id),
        getOrCreateRect: (id: string) => entries.get(id) ?? entryFor(),
        peek: (id: string) => entries.get(id),
      } as unknown as CreateCompositeFrameDeps['layerCache'],
      previews,
      raster: { beginFrame, enforceBudget: enforce },
      rasterizeLayer,
      stores,
      transformOverrides: overrides,
      viewport: {
        getDpr: () => 1,
        getViewportSize: () => ({ height: 64, width: 64 }),
        getZoom: () => 1,
        screenToDocument: ({ x, y }: { x: number; y: number }) => ({ x, y }),
      } as unknown as CreateCompositeFrameDeps['viewport'],
    },
    enforce,
    entries,
    overrides,
    previews,
    rasterizeLayer,
    stores,
  };
};

const draw = (
  doc = documentOf([layer('a')]),
  floatFrame: FloatingSelectionFrame | null = null,
  sam: SamPreviewState | null = null
) => createCompositeFrame(harness.deps).draw(surface(), doc, VIEW, floatFrame, sam);

/** The options object the compositor was handed on the most recent draw. */
const lastCompositeOptions = () => composite.mock.calls.at(-1)?.[4] as Record<string, unknown>;

beforeEach(() => {
  harness = makeHarness();
  composite = vi.spyOn(compositor, 'compositeDocument').mockImplementation(() => undefined);
});

describe('cache preparation', () => {
  it('starts a rasterization for a stale layer the frame demands', () => {
    harness.entries.set('a', entryFor(undefined, true));
    draw();
    expect(harness.rasterizeLayer).toHaveBeenCalledWith(expect.objectContaining({ id: 'a' }), expect.anything());
  });

  it('leaves a fresh cache alone', () => {
    draw();
    expect(harness.rasterizeLayer).not.toHaveBeenCalled();
  });

  it('does not rasterize a disabled layer', () => {
    harness.entries.set('a', entryFor(undefined, true));
    draw(documentOf([layer('a', { isEnabled: false })]));
    expect(harness.rasterizeLayer).not.toHaveBeenCalled();
  });
});

describe('SAM isolation', () => {
  const isolated = (): SamPreviewState =>
    ({
      data: surface(8, 8),
      guard: { layerId: 'a' },
      isolated: true,
      rect: { height: 8, width: 8, x: 0, y: 0 },
    }) as unknown as SamPreviewState;

  it('isolates the layer through the composition plan', () => {
    draw(documentOf([layer('a'), layer('b')]), null, isolated());
    const doc = composite.mock.calls.at(-1)?.[1] as CanvasDocumentContractV3;
    expect(getDocumentLeaves(doc).map((entry) => entry.id)).toEqual(['a', 'b']);
    expect(lastCompositeOptions().isolationLayerId).toBe('a');
  });

  it('clips to the preview rect', () => {
    draw(documentOf([layer('a')]), null, isolated());
    expect(lastCompositeOptions().clipRect).toEqual({ height: 8, width: 8, x: 0, y: 0 });
  });

  it.each(['layerPreviews', 'stagedPreview', 'floatingSelection', 'transformOverrides'])(
    'suppresses %s so nothing but the isolated layer is judged',
    (field) => {
      harness.previews.publishFilter('a', harness.previews.beginGuardedFilter('a'), {
        guard: { layerId: 'a' },
        rect: { height: 8, width: 8, x: 0, y: 0 },
        surface: surface(8, 8),
      } as never);
      harness.previews.publishStaged(harness.previews.nextStagedToken(), {
        height: 8,
        surface: surface(8, 8),
        width: 8,
      } as never);
      harness.overrides.set('a', { x: 5, y: 5 });
      const float = { composite: { layerId: 'a' } } as unknown as FloatingSelectionFrame;
      draw(documentOf([layer('a')]), float, isolated());
      expect(lastCompositeOptions()[field]).toBeNull();
    }
  );

  it('draws the whole document when the SAM preview is not isolated', () => {
    const sam = { ...isolated(), isolated: false } as SamPreviewState;
    draw(documentOf([layer('a'), layer('b')]), null, sam);
    const doc = composite.mock.calls.at(-1)?.[1] as CanvasDocumentContractV3;
    expect(getDocumentLeaves(doc).map((entry) => entry.id)).toEqual(['a', 'b']);
    expect(lastCompositeOptions().clipRect).toBeNull();
  });
});

describe('filter previews', () => {
  it('allocates no snapshot when no layer carries one', () => {
    const snapshot = vi.spyOn(harness.previews, 'filterSnapshot');
    draw();
    expect(lastCompositeOptions().layerPreviews).toBeNull();
    // This runs every composite frame; snapshotting unconditionally would
    // allocate a map per frame for the overwhelmingly common case of none.
    expect(snapshot).not.toHaveBeenCalled();
  });

  it('passes them through when one is active', () => {
    harness.previews.publishFilter('a', harness.previews.beginGuardedFilter('a'), {
      guard: { layerId: 'a' },
      rect: { height: 8, width: 8, x: 0, y: 0 },
      surface: surface(8, 8),
    } as never);
    draw();
    expect(lastCompositeOptions().layerPreviews).toBeInstanceOf(Map);
    expect((lastCompositeOptions().layerPreviews as Map<string, unknown>).has('a')).toBe(true);
  });
});

describe('the staged candidate', () => {
  const publishStaged = (placement?: Record<string, number>) => {
    harness.previews.publishStaged(harness.previews.nextStagedToken(), {
      height: 16,
      placement,
      surface: surface(16, 16),
      width: 16,
    } as never);
  };

  it('follows the bbox origin when it carries no placement', () => {
    publishStaged();
    draw();
    expect(lastCompositeOptions().stagedPreview).toMatchObject({
      opacity: 1,
      rect: { height: 16, width: 16, x: 4, y: 6 },
    });
  });

  it('uses its own placement when it has one', () => {
    publishStaged({ height: 2, opacity: 0.5, width: 3, x: 20, y: 30 });
    draw();
    expect(lastCompositeOptions().stagedPreview).toMatchObject({
      opacity: 0.5,
      rect: { height: 2, width: 3, x: 20, y: 30 },
    });
  });
});

describe('compositor options', () => {
  it('omits the checkerboard while the toggle is off', () => {
    harness.stores.checkerboard.set(false);
    draw();
    expect(lastCompositeOptions().checkerboardTile).toBeNull();
  });

  it('skips a layer with an open text-edit session so the portal is not double-drawn', () => {
    harness.stores.textEditSession.set({ id: 1, layerId: 'a' } as never);
    draw();
    expect(lastCompositeOptions().skipLayerId).toBe('a');
  });

  it('passes transform overrides only when some exist', () => {
    draw();
    expect(lastCompositeOptions().transformOverrides).toBeNull();
    harness.overrides.set('a', { x: 1, y: 1 });
    draw();
    expect(lastCompositeOptions().transformOverrides).toBe(harness.overrides);
  });
});

describe('the surface budget', () => {
  const workingSet = () => harness.enforce.mock.calls.at(-1)![0] as Set<string>;

  it('keeps a filter-previewed layer resident while only its preview is on screen', () => {
    const offscreen = documentOf([layer('a', { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 500, y: 0 } })]);
    draw(offscreen);
    expect(workingSet().has('a')).toBe(false);

    harness.previews.publishFilter('a', harness.previews.beginGuardedFilter('a'), {
      guard: { layerId: 'a' },
      rect: { height: 8, width: 8, x: -500, y: 0 },
      surface: surface(8, 8),
    } as never);
    draw(offscreen);
    expect(workingSet().has('a')).toBe(true);
  });

  it('re-derives a document-sized layer when the document grows around it', () => {
    harness.entries.clear();
    const gradient = layer('g', {
      source: { angle: 0, kind: 'linear', stops: [], type: 'gradient' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: -100, y: 0 },
    });
    const frame = createCompositeFrame(harness.deps);
    frame.draw(surface(), documentOf([gradient]), VIEW, null, null);
    expect(workingSet().has('g')).toBe(false);

    const grown = { ...documentOf([gradient]), width: 200 } as CanvasDocumentContractV3;
    frame.draw(surface(), grown, VIEW, null, null);
    expect(workingSet().has('g')).toBe(true);
  });

  it('hands the raster owner the layers this frame demanded and the usage captured before drawing', () => {
    draw();
    const [workingSet, usage] = harness.enforce.mock.calls.at(-1) as [Set<string>, typeof FRAME_USAGE];
    expect(workingSet.has('a')).toBe(true);
    expect(usage).toBe(FRAME_USAGE);
    expect(harness.beginFrame.mock.invocationCallOrder[0]).toBeLessThan(composite.mock.invocationCallOrder[0]!);
  });

  it('prunes the thumbnail state of an evicted layer', () => {
    harness.stores.thumbnailVersion.set('gone', 4);
    harness.stores.thumbnailStatus.set('gone', 'ready');
    harness.enforce.mockReturnValue({ evictedBaseLayerIds: ['gone'], overageBytes: 0 });
    draw();
    expect(harness.stores.thumbnailVersion.get('gone')).toBeUndefined();
    expect(harness.stores.thumbnailStatus.get('gone')).toBeUndefined();
  });

  it('enforces the budget after compositing, never before', () => {
    draw();
    expect(composite.mock.invocationCallOrder[0]).toBeLessThan(harness.enforce.mock.invocationCallOrder[0]!);
  });
});

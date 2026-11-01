/**
 * Real-worker checks that generation and export composites draw exactly what the settled display shows for spatial
 * stacks, layers and groups alike, whatever the bbox.
 */

import type { CanvasAdjustmentsContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { GroupSurfaceResult } from '@workbench/canvas-engine/render/groupSurfaceCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { compositeDocument } from '@workbench/canvas-engine/render/compositor';
import { createGroupSurfaceCache } from '@workbench/canvas-engine/render/groupSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { planBaseRasterComposite, renderRasterComposite } from '@workbench/canvas-engine/render/rasterComposite';
import {
  createSpatialAdjustmentService,
  type SpatialAdjustmentService,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { createCompositeDedupeCache, executeCompositePlan } from '@workbench/canvas-operations/compositeForGeneration';
import { afterEach, describe, expect, it, vi } from 'vitest';

const WIDTH = 64;
const HEIGHT = 64;
const BBOX: Rect = { height: HEIGHT, width: WIDTH, x: 0, y: 0 };
const MEMBER: Rect = { height: 30, width: 40, x: 0, y: 0 };
const TRANSLATED = { rotation: 0, scaleX: 1, scaleY: 1, x: 9, y: 7 };
const ROTATED_DOWNSCALED = { rotation: Math.PI / 7, scaleX: 0.6, scaleY: 0.5, x: 22, y: 4 };

/** Per-pixel runs around two spatial runs; noise places its grain by pixel, so a misplaced origin shows. */
const STACK: CanvasAdjustmentsContract = [
  { brightness: 0.1, contrast: 0.05, id: 'bc', isEnabled: true, type: 'brightness-contrast' },
  { amount: 70, id: 'clarity', isEnabled: true, size: 12, type: 'clarity' },
  {
    amount: 0.15,
    distribution: 'gaussian',
    grain: 'color',
    id: 'grain',
    isEnabled: true,
    response: 'flat',
    seed: 7,
    size: 1,
    type: 'noise',
  },
  { amount: 60, id: 'texture', isEnabled: true, size: 4, type: 'texture' },
];
const GROUP_STACK: CanvasAdjustmentsContract = [
  { amount: 80, id: 'gclarity', isEnabled: true, size: 10, type: 'clarity' },
  { amount: 40, id: 'gdehaze', isEnabled: true, type: 'dehaze' },
];

const raster = (id: string, overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: { height: MEMBER.height, imageName: id, width: MEMBER.width }, type: 'paint' },
  transform: TRANSLATED,
  type: 'raster',
  ...overrides,
});

const docWith = (nodes: Parameters<typeof stacksFrom>[0]) => ({
  background: 'transparent' as const,
  bbox: BBOX,
  height: HEIGHT,
  selectedLayerId: null,
  stacks: stacksFrom(nodes),
  version: 3 as const,
  width: WIDTH,
});

const worstDifference = (a: Uint8ClampedArray, b: Uint8ClampedArray): number => {
  expect(a.length).toBe(b.length);
  let worst = 0;
  for (let i = 0; i < a.length; i += 1) {
    worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  }
  return worst;
};

const pixelsOf = (surface: RasterSurface, rect: Rect = { ...BBOX, height: surface.height, width: surface.width }) =>
  surface.ctx.getImageData(rect.x, rect.y, rect.width, rect.height).data;

const services: SpatialAdjustmentService[] = [];
afterEach(() => {
  for (const service of services.splice(0)) {
    service.dispose();
  }
});

/** Layers `ids` hold a gradient with a transparent border and a dark block: edges for the kernels, and a sub-domain. */
const sceneFor = (ids: readonly string[]) => {
  const backend = createDomRasterBackend();
  const caches = createLayerCacheStore(backend);
  for (const [index, id] of ids.entries()) {
    const ctx = caches.growToRect(id, MEMBER).surface.ctx;
    const gradient = ctx.createLinearGradient(0, 0, MEMBER.width, MEMBER.height);
    gradient.addColorStop(0, index === 0 ? '#1d4ed8' : '#15803d');
    gradient.addColorStop(0.5, '#e2e8f0');
    gradient.addColorStop(1, '#f59e0b');
    ctx.fillStyle = gradient;
    ctx.fillRect(3, 2, MEMBER.width - 7, MEMBER.height - 5);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(14, 9, 10, 8);
    caches.publishPixels(id);
  }
  const memory = new RasterMemoryBudgetController({ budgetBytes: 64 * 1024 * 1024 });
  const service = createSpatialAdjustmentService({
    reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'spatial-adjustment' }),
  });
  services.push(service);
  const getLayerSurface = (layerId: string) => {
    const entry = caches.get(layerId)!;
    return Promise.resolve({ rect: entry.rect, release: () => undefined, surface: entry.surface });
  };
  /** The export composite: no display copy to reuse, so every spatial stack runs a job. */
  const exported = (document: ReturnType<typeof docWith>, bbox: Rect) =>
    renderRasterComposite(planBaseRasterComposite(document, bbox), {
      backend,
      getLayerSurface,
      runSpatial: (request) => service.run(request),
    });
  return { backend, caches, exported, getLayerSurface, memory, service };
};

/** Composites until `done` holds, as frames would while results land. */
const settle = async (composite: () => void, done: () => boolean) => {
  await vi.waitFor(
    () => {
      composite();
      expect(done()).toBe(true);
    },
    { interval: 20, timeout: 10_000 }
  );
};

/** The display of a document whose layers carry their own stacks, once every spatial result has landed. */
const settledLayerDisplay = async (scene: ReturnType<typeof sceneFor>, document: ReturnType<typeof docWith>) => {
  const display = createAdjustedSurfaceCache(scene.backend, undefined, {
    spatial: { run: (request) => scene.service.run(request) },
  });
  const layers = document.stacks.raster as CanvasRasterLayerContractV2[];
  const screen = scene.backend.createSurface(WIDTH, HEIGHT);
  const composite = () => {
    screen.ctx.clearRect(0, 0, WIDTH, HEIGHT);
    compositeDocument(screen, document, scene.caches, identity(), {
      adjustedSurface: (layer, entry) =>
        layer.type === 'raster' ? display.get(layer.id, entry, layer.adjustments) : null,
      backend: scene.backend,
    });
  };
  await settle(composite, () =>
    layers.every((layer) => display.peek(layer.id, scene.caches.get(layer.id)!, layer.adjustments) !== null)
  );
  return { display, screen };
};

/** The display of spatial groups over raw members, once the outermost group's pixels are exact. */
const settledGroupDisplay = async (
  scene: ReturnType<typeof sceneFor>,
  document: ReturnType<typeof docWith>,
  outermost: string,
  inner: readonly string[] = []
) => {
  const groupSurfaces = createGroupSurfaceCache({
    createSurface: (w, h) => scene.backend.createSurface(w, h),
    damageSince: (id, version) => scene.caches.damageSince(id, version),
    getAdjustedSurface: () => ({ state: 'exact', surface: null }),
    getCacheEntry: (id) => scene.caches.get(id),
    spatial: {
      // Wired as the engine does: a landing redraws the groups holding the owner.
      onSettle: (groupId) => {
        if (inner.includes(groupId)) {
          groupSurfaces.invalidate([outermost]);
        }
      },
      run: (request) => scene.service.run(request),
    },
  });
  const screen = scene.backend.createSurface(WIDTH, HEIGHT);
  let state: GroupSurfaceResult['state'] | undefined;
  const composite = () => {
    screen.ctx.clearRect(0, 0, WIDTH, HEIGHT);
    compositeDocument(screen, document, scene.caches, identity(), {
      backend: scene.backend,
      groupSurface: (scope, members, matrices, content) => {
        const result = groupSurfaces.get(scope, members, matrices, content);
        state = scope.id === outermost ? result?.state : state;
        return result;
      },
    });
  };
  await settle(composite, () => state === 'exact');
  return screen;
};

describe('spatial layer composites', () => {
  it.each([
    ['translated', TRANSLATED],
    ['rotated and downscaled', ROTATED_DOWNSCALED],
  ])('exports a %s layer as the settled display shows it', async (_, transform) => {
    const scene = sceneFor(['m']);
    const layer = raster('m', { adjustments: STACK, transform });
    const document = docWith([layer]);
    const { display, screen } = await settledLayerDisplay(scene, document);

    const fromJob = await scene.exported(document, BBOX);
    const fromDisplayCopy = await renderRasterComposite(planBaseRasterComposite(document, BBOX), {
      adjustedSurface: (layerId, surface, adjustments) => {
        const entry = scene.caches.get(layerId)!;
        return entry.surface === surface ? display.peek(layerId, entry, adjustments) : null;
      },
      backend: scene.backend,
      getLayerSurface: scene.getLayerSurface,
    });

    expect(worstDifference(pixelsOf(fromJob), pixelsOf(screen))).toBeLessThanOrEqual(1);
    expect(worstDifference(pixelsOf(fromDisplayCopy), pixelsOf(screen))).toBe(0);
    // Leaving the spatial entries out would be visible: the per-pixel interim differs from what landed.
    const interim = scene.backend.createSurface(WIDTH, HEIGHT);
    const perPixel = createAdjustedSurfaceCache(scene.backend);
    compositeDocument(interim, document, scene.caches, identity(), {
      adjustedSurface: (_layer, entry) => perPixel.get('m', entry, STACK),
      backend: scene.backend,
    });
    expect(worstDifference(pixelsOf(fromJob), pixelsOf(interim))).toBeGreaterThan(8);
    expect(scene.memory.snapshot().reservedBytes).toBe(0);
  });

  it.each([
    ['translated', TRANSLATED],
    ['rotated and downscaled', ROTATED_DOWNSCALED],
  ])('draws a %s layer identically over any bbox that crops it', async (_, transform) => {
    const scene = sceneFor(['m']);
    const document = docWith([raster('m', { adjustments: STACK, transform })]);
    const crop: Rect = { height: 14, width: 19, x: 17, y: 12 };

    const whole = await scene.exported(document, BBOX);
    const cropped = await scene.exported(document, crop);

    expect(worstDifference(pixelsOf(cropped), pixelsOf(whole, crop))).toBe(0);
  });
});

describe('spatial group composites', () => {
  /** Two members, one partly outside a cropping bbox, under a spatial group. */
  const groupDocument = () =>
    docWith([
      groupContract('g', [raster('m'), raster('n', { opacity: 0.7, transform: { ...TRANSLATED, x: 21, y: 26 } })], {
        adjustments: GROUP_STACK,
      }),
    ]);

  it('exports a group as the settled display shows it: its stack on the union of its members', async () => {
    const scene = sceneFor(['m', 'n']);
    const document = groupDocument();
    const screen = await settledGroupDisplay(scene, document, 'g');

    expect(worstDifference(pixelsOf(await scene.exported(document, BBOX)), pixelsOf(screen))).toBe(0);
  });

  it('draws a group identically over any bbox that crops it', async () => {
    const scene = sceneFor(['m', 'n']);
    const document = groupDocument();
    const crop: Rect = { height: 20, width: 24, x: 18, y: 20 };

    const whole = await scene.exported(document, BBOX);
    const cropped = await scene.exported(document, crop);

    expect(worstDifference(pixelsOf(cropped), pixelsOf(whole, crop))).toBe(0);
  });

  it('exports nested spatial groups as the settled display shows them', async () => {
    const scene = sceneFor(['m', 'n']);
    const document = docWith([
      groupContract(
        'outer',
        [
          groupContract('inner', [raster('m')], {
            adjustments: [{ amount: 90, id: 'texture', isEnabled: true, size: 4, type: 'texture' }],
          }),
          raster('n', { transform: { ...TRANSLATED, x: 24, y: 30 } }),
        ],
        { adjustments: GROUP_STACK }
      ),
    ]);
    const screen = await settledGroupDisplay(scene, document, 'outer', ['inner']);

    expect(worstDifference(pixelsOf(await scene.exported(document, BBOX)), pixelsOf(screen))).toBe(0);
  });
});

describe('generation composite', () => {
  it('uploads the pixels the export composite draws', async () => {
    const scene = sceneFor(['m', 'n']);
    const document = docWith([
      raster('m', { adjustments: STACK }),
      groupContract('g', [raster('n', { transform: { ...TRANSLATED, x: 20, y: 24 } })], { adjustments: GROUP_STACK }),
    ]);
    const entry = planBaseRasterComposite(document, BBOX);
    let uploaded: Blob | null = null;

    await executeCompositePlan(
      { bbox: BBOX, entries: [entry] },
      {
        backend: {
          createSurface: (width, height) => scene.backend.createSurface(width, height),
          encodeSurface: (surface) => scene.backend.encodeSurface(surface, 'image/png'),
        },
        dedupe: createCompositeDedupeCache(),
        // Generation composites detached copies, never the live caches the display adjusts.
        getLayerSurface: async (layerId) => {
          const live = await scene.getLayerSurface(layerId);
          const detached = scene.backend.createSurface(live.rect.width, live.rect.height);
          detached.ctx.drawImage(live.surface.canvas, 0, 0);
          return { ...live, surface: detached };
        },
        runSpatial: (request) => scene.service.run(request),
        uploadImage: (blob) => {
          uploaded = blob;
          return Promise.resolve({ height: HEIGHT, imageName: 'composite.png', width: WIDTH });
        },
      }
    );

    const decoded = scene.backend.createSurface(WIDTH, HEIGHT);
    decoded.ctx.drawImage(await createImageBitmap(uploaded!), 0, 0);
    expect(worstDifference(pixelsOf(decoded), pixelsOf(await scene.exported(document, BBOX)))).toBe(0);
  });
});

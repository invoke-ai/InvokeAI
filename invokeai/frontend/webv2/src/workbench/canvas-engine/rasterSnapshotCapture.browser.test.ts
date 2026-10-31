import {
  createCompositeDedupeCache,
  executeControlComposite,
  executeMaskComposite,
  executeRegionalMaskComposite,
} from '@workbench/canvas-operations/compositeForGeneration';
import {
  planComposites,
  planControlComposites,
  planRegionalMaskComposites,
} from '@workbench/canvas-operations/generationCompositePlan';
import { describe, expect, it } from 'vitest';

import type { LayerExportGuard } from './capabilities';
import type {
  CanvasAdjustmentsContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasStateContractV3,
} from './contracts';
import type { CanvasRasterSnapshot } from './rasterTransactions';
import type { Rect } from './types';

import { RasterMemoryBudgetController } from './controllers/rasterMemoryBudgetController';
import { groupContract, stacksFrom } from './document-model/documentFixtures.testStub';
import { createRasterSnapshotCapture } from './rasterSnapshotCapture';
import { createDomRasterBackend } from './render/raster';
import { planBaseRasterComposite, renderRasterComposite } from './render/rasterComposite';
import { createSpatialAdjustmentService } from './render/spatial/spatialAdjustmentService';

const noise: CanvasAdjustmentsContract = [
  {
    amount: 0.15,
    distribution: 'gaussian',
    grain: 'color',
    id: 'noise',
    isEnabled: true,
    response: 'flat',
    seed: 7,
    size: 3,
    type: 'noise',
  },
];
const color: CanvasAdjustmentsContract = [
  {
    brightness: 0.1,
    contrast: 0.1,
    id: 'color',
    isEnabled: true,
    type: 'brightness-contrast',
  },
];
const spatial: CanvasAdjustmentsContract = [
  {
    amount: 70,
    id: 'clarity',
    isEnabled: true,
    size: 12,
    type: 'clarity',
  },
];
const identity = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };
const transforms = [
  ['translated', { ...identity, x: 13.25, y: -7.5 }],
  ['rotated', { ...identity, rotation: Math.PI / 7, x: 45, y: -3 }],
  ['flipped', { ...identity, scaleX: -1.2, scaleY: 0.8, x: 90, y: 3 }],
  ['downscaled', { ...identity, scaleX: 0.2, scaleY: 0.3, x: 30, y: 15 }],
] as const;
const origin = { x: -19, y: -11 };
const content = { ...origin, width: 240, height: 180 };
const bbox: Rect = { x: 37, y: 24, width: 18, height: 20 };

const scene = (
  transform = identity,
  adjustments?: CanvasAdjustmentsContract,
  groupAdjustments?: CanvasAdjustmentsContract,
  sourceRect = content
) => {
  const backend = createDomRasterBackend();
  const surface = backend.createSurface(content.width, content.height);
  const pixels = surface.ctx.createImageData(content.width, content.height);
  for (let y = 0; y < content.height; y++) {
    for (let x = 0; x < content.width; x++) {
      const i = (y * content.width + x) * 4;
      pixels.data.set(
        [(x * 17 + y * 11) % 256, (x * 3 + y * 7) % 256, (x * 5 + y * 13) % 256, (x + y) % 7 === 0 ? 100 : 255],
        i
      );
    }
  }
  surface.ctx.putImageData(pixels, 0, 0);
  const layer: CanvasRasterLayerContractV2 = {
    id: 'map',
    name: 'Map',
    type: 'raster',
    isEnabled: true,
    isLocked: false,
    blendMode: 'normal',
    opacity: 0.8,
    source: {
      type: 'paint',
      bitmap: { width: content.width, height: content.height, imageName: 'map.png' },
      offset: { x: sourceRect.x, y: sourceRect.y },
    },
    transform,
    adjustments,
  };
  const document = {
    background: 'transparent' as const,
    bbox,
    width: 256,
    height: 256,
    selectedLayerId: null,
    version: 3 as const,
    stacks: stacksFrom(
      groupAdjustments ? [groupContract('group', [layer], { adjustments: groupAdjustments })] : [layer]
    ),
  };
  const canvas = { document } as CanvasStateContractV3;
  const memory = new RasterMemoryBudgetController();
  const capture = createRasterSnapshotCapture({
    memory,
    createSurface: backend.createSurface,
    getCanvasState: () => canvas,
    getDirectPixelEpoch: () => 0,
    getDocumentGeneration: () => 0,
    getLifecycleGeneration: () => 0,
    isDisposed: () => false,
    isGuardCurrent: () => true,
    rasterizeLayerPixels: () =>
      Promise.resolve({
        status: 'ok',
        surface,
        rect: sourceRect,
        guard: { layerId: layer.id } as LayerExportGuard,
        release: () => undefined,
      }),
  });
  const service = createSpatialAdjustmentService({
    reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'spatial-adjustment' }),
  });
  const render = (snapshot: CanvasRasterSnapshot) =>
    renderRasterComposite(planBaseRasterComposite(document, bbox), {
      backend,
      getLayerSurface: (id) => Promise.resolve({ ...snapshot.layerSurfaces.get(id)!, release: () => undefined }),
      runSpatial: (request) => service.run(request),
    });
  return { capture, surface, render, service, memory, canvas, backend, layer, content: sourceRect };
};

const compare = async (s: ReturnType<typeof scene>, fullSource: boolean) => {
  const { content } = s;
  const document = s.capture.captureDocumentSnapshot()!;
  const full = await s.capture.captureRasterSnapshot(document, ['map']);
  const cropped = await s.capture.captureRasterSnapshot(document, ['map'], { region: bbox });
  expect(full.status).toBe('ok');
  expect(cropped.status).toBe('ok');
  if (full.status !== 'ok' || cropped.status !== 'ok') {
    throw new Error('capture failed');
  }
  try {
    const detached = cropped.snapshot.layerSurfaces.get('map')!;
    expect(detached.surface).not.toBe(s.surface);
    expect(detached.sourceRect).toEqual(content);
    if (fullSource) {
      expect(detached.rect).toEqual(content);
    } else {
      expect(detached.surface.width * detached.surface.height).toBeLessThan(content.width * content.height);
    }
    // Edits after detachment must not alter either frozen generation input.
    s.surface.ctx.clearRect(0, 0, content.width, content.height);
    const a = await s.render(full.snapshot);
    const b = await s.render(cropped.snapshot);
    const expected = a.ctx.getImageData(0, 0, bbox.width, bbox.height).data;
    const actual = b.ctx.getImageData(0, 0, bbox.width, bbox.height).data;
    expect(expected.some((value) => value !== 0)).toBe(true);
    expect(actual).toEqual(expected);
  } finally {
    full.snapshot.release();
    cropped.snapshot.release();
    s.service.dispose();
  }
  expect(s.memory.snapshot()).toMatchObject({ reservedBytes: 0, detachedBytes: 0 });
};

describe('region snapshots through real Canvas2D compositing', () => {
  it.each(transforms)('preserves %s pixels from a nonzero paint origin', async (_name, transform) => {
    await compare(
      scene(transform),
      transform.rotation !== 0 || Math.abs(transform.scaleX) !== 1 || Math.abs(transform.scaleY) !== 1
    );
  });
  it.each([
    ['color', color],
    ['noise', noise],
  ] as const)('preserves %s adjustment sampling', async (_name, stack) => {
    await compare(scene(identity, [...stack]), false);
    await compare(scene(transforms[3][1], [...stack]), true);
  });
  it('preserves layer noise inside an adjusted group', async () => {
    await compare(scene(identity, noise, color), false);
  });
  it.each([-2, -1.2, -1, -0.7, 0.2, 0.3, 0.8, 1, 1.2, 2])('keeps source sampling at scale %s', async (scale) => {
    const t = { rotation: 0, scaleX: scale, scaleY: 0.7, x: bbox.x - 60 * scale, y: bbox.y - 30 };
    await compare(scene(t), true);
    await compare(scene(t, color), true);
  });
  it.each([-Math.PI / 3, -Math.PI / 8, 0, Math.PI / 8, Math.PI / 3, Math.PI / 2])(
    'keeps rotated sampling on its original source at %s',
    async (rotation) => {
      await compare(
        scene({
          rotation,
          scaleX: -1,
          scaleY: 1,
          x: 46 + 60 * Math.cos(rotation) + 50 * Math.sin(rotation),
          y: 34 + 60 * Math.sin(rotation) - 50 * Math.cos(rotation),
        }),
        rotation !== 0
      );
    }
  );
  it.each([0, 0.1, 0.2, 0.5, 0.7, 0.99])('crops fractional translation %s with either reflection', async (fraction) => {
    for (const scaleX of [-1, 1]) {
      for (const scaleY of [-1, 1]) {
        await compare(
          scene({ rotation: 0, scaleX, scaleY, x: 46 - 60 * scaleX + fraction, y: 34 - 50 * scaleY + fraction }),
          false
        );
      }
    }
  });
  it.each([
    { x: -19.4, y: -11 },
    { x: -19, y: -11.6 },
    { x: -19.4, y: -11.6 },
  ])('preserves fractional source origin $x, $y through sampled adjustments', async (offset) => {
    for (const stack of [undefined, color, noise]) {
      await compare(scene(transforms[0][1], stack, undefined, { ...content, ...offset }), false);
      await compare(
        scene({ rotation: 0, scaleX: -1, scaleY: -1, x: 107.2, y: 75.7 }, stack, undefined, { ...content, ...offset }),
        false
      );
    }
  });
  it('keeps a spatial layer stack on its full source', async () => {
    await compare(scene(identity, [...noise, ...spatial, ...color]), true);
  });
  it('keeps members of a spatial group on their full sources', async () => {
    await compare(scene(identity, noise, spatial), true);
  });
});

it.each([origin, { x: -19.4, y: -11.6 }])(
  'keeps control, regional and inpaint-mask uploads pixel-identical at $x, $y',
  async (offset) => {
    const s = scene({ rotation: 0, scaleX: -1, scaleY: 1, x: 107.2, y: -15.7 }, undefined, undefined, {
      ...content,
      ...offset,
    });
    const common = {
      isEnabled: true,
      isLocked: false,
      blendMode: 'normal' as const,
      opacity: 1,
      transform: s.layer.transform,
    };
    const mask = {
      bitmap: { width: content.width, height: content.height, imageName: 'mask.png' },
      offset,
      fill: { color: '#ff0000', style: 'solid' as const },
    };
    const layers: CanvasLayerContract[] = [
      { ...common, type: 'inpaint_mask', id: 'mask', name: 'Mask', mask },
      {
        ...common,
        type: 'regional_guidance',
        id: 'region',
        name: 'Region',
        mask,
        autoNegative: false,
        positivePrompt: 'region',
        negativePrompt: null,
        referenceImages: [],
      },
      {
        ...common,
        type: 'control',
        id: 'control',
        name: 'Control',
        source: s.layer.source,
        withTransparencyEffect: true,
        adapter: { kind: 'controlnet', model: null, beginEndStepPct: [0, 1], controlMode: 'balanced', weight: 1 },
      },
    ];
    s.canvas.document.stacks = stacksFrom(layers);
    const document = s.capture.captureDocumentSnapshot()!;
    const full = await s.capture.captureRasterSnapshot(
      document,
      layers.map((layer) => layer.id)
    );
    const cropped = await s.capture.captureRasterSnapshot(
      document,
      layers.map((layer) => layer.id),
      { region: bbox }
    );
    if (full.status !== 'ok' || cropped.status !== 'ok') {
      throw new Error('capture failed');
    }
    const execute = async (snapshot: CanvasRasterSnapshot) => {
      const uploads: Uint8ClampedArray[] = [];
      const deps = {
        backend: s.backend,
        dedupe: createCompositeDedupeCache(),
        getLayerSurface: (id: string) =>
          Promise.resolve({ ...snapshot.layerSurfaces.get(id)!, release: () => undefined }),
        hashBlob: () => Promise.resolve(String(uploads.length)),
        uploadImage: async (blob: Blob) => {
          const bitmap = await s.backend.createImageBitmap(blob);
          const decoded = s.backend.createSurface(bbox.width, bbox.height);
          decoded.ctx.drawImage(bitmap, 0, 0);
          bitmap.close();
          uploads.push(decoded.ctx.getImageData(0, 0, bbox.width, bbox.height).data);
          return { imageName: String(uploads.length), width: bbox.width, height: bbox.height };
        },
      };
      await executeControlComposite(planControlComposites(document.canvas.document, bbox)[0]!.entry, deps);
      await executeRegionalMaskComposite(planRegionalMaskComposites(document.canvas.document, bbox)[0]!.entry, deps);
      await executeMaskComposite(
        planComposites(document.canvas.document, bbox).entries.find((entry) => entry.kind === 'inpaint-mask')!,
        deps
      );
      return uploads;
    };
    try {
      s.surface.ctx.clearRect(0, 0, content.width, content.height);
      const expected = await execute(full.snapshot);
      const actual = await execute(cropped.snapshot);
      expect(expected).toHaveLength(3);
      expect(actual).toEqual(expected);
    } finally {
      full.snapshot.release();
      cropped.snapshot.release();
      s.service.dispose();
    }
    expect(s.memory.snapshot()).toMatchObject({ reservedBytes: 0, detachedBytes: 0 });
  }
);

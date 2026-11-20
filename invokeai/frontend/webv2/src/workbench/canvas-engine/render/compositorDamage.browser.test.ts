import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { LayerDamage, Mat2d, Rect } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { identity } from '@workbench/canvas-engine/math/mat2d';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

import { compositeDocument, createCheckerboardTile } from './compositor';

const SCREEN_W = 480;
const SCREEN_H = 360;

const rasterLayer = (id: string, overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 10, imageName: id, width: 10 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const scene = (layers: CanvasLayerContract[], rects: Record<string, Rect>, colors: Record<string, string>) => {
  const backend = createDomRasterBackend();
  const caches = createLayerCacheStore(backend);
  for (const layer of layers) {
    const rect = rects[layer.id]!;
    caches.growToRect(layer.id, rect);
    const entry = caches.get(layer.id)!;
    entry.surface.ctx.fillStyle = colors[layer.id]!;
    entry.surface.ctx.fillRect(0, 0, rect.width, rect.height);
    caches.publishPixels(layer.id);
  }
  return { backend, caches };
};

const repaint = (
  caches: ReturnType<typeof createLayerCacheStore>,
  layerId: string,
  rect: Rect,
  color: string
): LayerDamage => {
  const entry = caches.get(layerId)!;
  const ctx = entry.surface.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = color;
  ctx.fillRect(rect.x - entry.rect.x, rect.y - entry.rect.y, rect.width, rect.height);
  caches.publishPixels(layerId);
  return { layerId, rect };
};

const pixelsOf = (surface: { ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D }) =>
  surface.ctx.getImageData(0, 0, SCREEN_W, SCREEN_H).data;

const countDifferences = (a: Uint8ClampedArray, b: Uint8ClampedArray): { differing: number; worst: number } => {
  let differing = 0;
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const delta = Math.abs(a[i]! - b[i]!);
    if (delta > 0) {
      differing++;
      worst = Math.max(worst, delta);
    }
  }
  return { differing, worst };
};

describe('damage-clipped compositing matches a full repaint', () => {
  const run = (
    doc: CanvasDocumentContractV3,
    caches: ReturnType<typeof createLayerCacheStore>,
    backend: ReturnType<typeof createDomRasterBackend>,
    view: Mat2d,
    steps: (() => LayerDamage)[],
    checkerboard: boolean,
    extra: Partial<Parameters<typeof compositeDocument>[4]> = {}
  ) => {
    const full = backend.createSurface(SCREEN_W, SCREEN_H);
    const clipped = backend.createSurface(SCREEN_W, SCREEN_H);
    const tile = checkerboard ? createCheckerboardTile(backend, { a: '#2a2a2a', b: '#363636' }) : null;
    const opts = { backend, checkerboardTile: tile, imageSmoothing: false, ...extra };

    compositeDocument(full, doc, caches, view, opts);
    compositeDocument(clipped, doc, caches, view, opts);

    for (const step of steps) {
      const damage = step();
      compositeDocument(full, doc, caches, view, opts);
      compositeDocument(clipped, doc, caches, view, {
        ...opts,
        damage: { kind: 'regions', regions: [damage] } as const,
      });
    }
    return { clipped: pixelsOf(clipped), full: pixelsOf(full) };
  };

  const IDENTITY_VIEW = identity();

  it('tracks repeated edits to one layer', () => {
    const layers = [rasterLayer('a')];
    const rects = { a: { height: 200, width: 300, x: 40, y: 30 } };
    const { backend, caches } = scene(layers, rects, { a: '#3b82f6' });
    const doc = { stacks: stacksFrom(layers) } as unknown as CanvasDocumentContractV3;
    const steps = Array.from(
      { length: 6 },
      (_, i) => () => repaint(caches, 'a', { height: 30, width: 40, x: 60 + i * 30, y: 60 + i * 20 }, '#ef4444')
    );
    const { clipped, full } = run(doc, caches, backend, IDENTITY_VIEW, steps, true);
    expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
  });

  it('tracks an edit under a rotated layer transform', () => {
    const layers = [rasterLayer('a', { transform: { rotation: 0.4, scaleX: 1, scaleY: 1, x: 90, y: 60 } })];
    const rects = { a: { height: 150, width: 200, x: 0, y: 0 } };
    const { backend, caches } = scene(layers, rects, { a: '#06b6d4' });
    const doc = { stacks: stacksFrom(layers) } as unknown as CanvasDocumentContractV3;
    const steps = [() => repaint(caches, 'a', { height: 40, width: 60, x: 30, y: 40 }, '#f97316')];
    const { clipped, full } = run(doc, caches, backend, IDENTITY_VIEW, steps, true);
    expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
  });

  it('repaints in full when the damage names a layer that is gone', () => {
    const layers = [rasterLayer('a')];
    const rects = { a: { height: 200, width: 300, x: 40, y: 30 } };
    const { backend, caches } = scene(layers, rects, { a: '#3b82f6' });
    const doc = { stacks: stacksFrom(layers) } as unknown as CanvasDocumentContractV3;
    const steps = [
      () => {
        repaint(caches, 'a', { height: 80, width: 90, x: 60, y: 60 }, '#ef4444');
        return { layerId: 'nonexistent', rect: { height: 1, width: 1, x: 0, y: 0 } };
      },
    ];
    const { clipped, full } = run(doc, caches, backend, IDENTITY_VIEW, steps, true);
    expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
  });
});

describe('overlay stacks switched off while a result is staged', () => {
  it('draws the raster under a mask, a region and a regenerate region untinted, and tinted again after', () => {
    const mask: CanvasLayerContract = {
      blendMode: 'normal',
      id: 'mask',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: null, fill: { color: '#0000ff', style: 'solid' } },
      name: 'mask',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    };
    const region = rasterLayer('region', { inpaint: { fill: { color: '#00ff00', style: 'solid' }, isEnabled: true } });
    const layers = [mask, region, rasterLayer('base')];
    const rects = {
      base: { height: 100, width: 300, x: 0, y: 0 },
      mask: { height: 100, width: 100, x: 0, y: 0 },
      region: { height: 100, width: 100, x: 200, y: 0 },
    };
    const { backend, caches } = scene(layers, rects, { base: '#ff0000', mask: '#ffffff', region: '#ff0000' });
    const doc = { stacks: stacksFrom(layers) } as unknown as CanvasDocumentContractV3;
    const screen = backend.createSurface(SCREEN_W, SCREEN_H);
    const rgbAt = (x: number): number[] => Array.from(screen.ctx.getImageData(x, 50, 1, 1).data.slice(0, 3));
    const draw = (staging: boolean) =>
      compositeDocument(screen, doc, caches, identity(), {
        backend,
        imageSmoothing: false,
        regionOverlays: !staging,
        showOverlayStacks: staging ? { control: false, inpaint_mask: false, regional_guidance: false } : undefined,
      });

    draw(true);
    expect([rgbAt(50), rgbAt(150), rgbAt(250)]).toEqual([
      [255, 0, 0],
      [255, 0, 0],
      [255, 0, 0],
    ]);

    draw(false);
    expect(rgbAt(50)).not.toEqual([255, 0, 0]);
    expect(rgbAt(150)).toEqual([255, 0, 0]);
    expect(rgbAt(250)).not.toEqual([255, 0, 0]);
  });
});

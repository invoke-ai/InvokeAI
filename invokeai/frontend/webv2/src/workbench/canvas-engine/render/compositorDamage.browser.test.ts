/**
 * The correctness guard for damage-clipped compositing.
 *
 * A damage-clipped frame leaves everything outside the damage rect showing the
 * PREVIOUS frame. That is the whole point, and also the whole risk: if the
 * damage under-reports by even a pixel, the target keeps stale content, and no
 * assertion about draw-call order would notice. So these tests run every case
 * twice against real canvases — once repainting in full, once damage-clipped
 * on top of the previous frame — and demand the two are pixel-identical.
 */

import type {
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { LayerDamage, Mat2d, Rect } from '@workbench/canvas-engine/types';

import { fromTRS, identity } from '@workbench/canvas-engine/math/mat2d';
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

/** A fresh backend + cache with each layer's surface painted a flat colour. */
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

/** Paints `rect` (layer-local) of a layer's cache a new colour. */
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
  /**
   * Composites `steps` twice: once always in full, once with each step's damage
   * clipped on top of the previous frame. Returns both screens' pixels.
   */
  const run = (
    doc: CanvasDocumentContractV2,
    caches: ReturnType<typeof createLayerCacheStore>,
    backend: ReturnType<typeof createDomRasterBackend>,
    view: Mat2d,
    steps: (() => LayerDamage)[],
    checkerboard: boolean
  ) => {
    const full = backend.createSurface(SCREEN_W, SCREEN_H);
    const clipped = backend.createSurface(SCREEN_W, SCREEN_H);
    const tile = checkerboard ? createCheckerboardTile(backend, { a: '#2a2a2a', b: '#363636' }) : null;
    const opts = { checkerboardTile: tile, imageSmoothing: false };

    // Seed both with an identical full composite.
    compositeDocument(full, doc, caches, view, opts);
    compositeDocument(clipped, doc, caches, view, opts);

    for (const step of steps) {
      const damage = step();
      compositeDocument(full, doc, caches, view, opts);
      compositeDocument(clipped, doc, caches, view, { ...opts, damage: [damage] });
    }
    return { clipped: pixelsOf(clipped), full: pixelsOf(full) };
  };

  const IDENTITY_VIEW = identity();

  it('tracks repeated edits to one layer', () => {
    const layers = [rasterLayer('a')];
    const rects = { a: { height: 200, width: 300, x: 40, y: 30 } };
    const { backend, caches } = scene(layers, rects, { a: '#3b82f6' });
    const doc = { layers } as unknown as CanvasDocumentContractV2;
    const steps = Array.from(
      { length: 6 },
      (_, i) => () => repaint(caches, 'a', { height: 30, width: 40, x: 60 + i * 30, y: 60 + i * 20 }, '#ef4444')
    );
    const { clipped, full } = run(doc, caches, backend, IDENTITY_VIEW, steps, true);
    expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
  });

  it('tracks an edit under a translated + scaled layer transform', () => {
    // The damage arrives layer-local, so the compositor has to carry it through
    // the layer matrix. A wrong mapping leaves the repainted area stale.
    const layers = [rasterLayer('a', { transform: { rotation: 0, scaleX: 1.7, scaleY: 1.3, x: 55, y: 25 } })];
    const rects = { a: { height: 160, width: 220, x: 0, y: 0 } };
    const { backend, caches } = scene(layers, rects, { a: '#22c55e' });
    const doc = { layers } as unknown as CanvasDocumentContractV2;
    const steps = [
      () => repaint(caches, 'a', { height: 40, width: 50, x: 20, y: 20 }, '#eab308'),
      () => repaint(caches, 'a', { height: 25, width: 25, x: 120, y: 90 }, '#a855f7'),
    ];
    const { clipped, full } = run(doc, caches, backend, IDENTITY_VIEW, steps, true);
    expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
  });

  it('tracks an edit under a rotated layer transform', () => {
    const layers = [rasterLayer('a', { transform: { rotation: 0.4, scaleX: 1, scaleY: 1, x: 90, y: 60 } })];
    const rects = { a: { height: 150, width: 200, x: 0, y: 0 } };
    const { backend, caches } = scene(layers, rects, { a: '#06b6d4' });
    const doc = { layers } as unknown as CanvasDocumentContractV2;
    const steps = [() => repaint(caches, 'a', { height: 40, width: 60, x: 30, y: 40 }, '#f97316')];
    const { clipped, full } = run(doc, caches, backend, IDENTITY_VIEW, steps, true);
    expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
  });

  it('tracks an edit to a lower layer with others stacked over it', () => {
    // The damaged layer is overdrawn by a translucent one above, so the clipped
    // frame has to redraw the whole stack inside the rect, not just that layer.
    const layers = [rasterLayer('top', { opacity: 0.5 }), rasterLayer('bottom')];
    const rects = {
      bottom: { height: 220, width: 320, x: 20, y: 20 },
      top: { height: 140, width: 180, x: 90, y: 70 },
    };
    const { backend, caches } = scene(layers, rects, { bottom: '#3b82f6', top: '#f43f5e' });
    const doc = { layers } as unknown as CanvasDocumentContractV2;
    const steps = [
      () => repaint(caches, 'bottom', { height: 60, width: 60, x: 100, y: 80 }, '#facc15'),
      () => repaint(caches, 'bottom', { height: 40, width: 200, x: 40, y: 200 }, '#14b8a6'),
    ];
    const { clipped, full } = run(doc, caches, backend, IDENTITY_VIEW, steps, true);
    expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
  });

  it('tracks edits under a zoomed and panned viewport', () => {
    const layers = [rasterLayer('a')];
    const rects = { a: { height: 180, width: 240, x: 10, y: 10 } };
    const { backend, caches } = scene(layers, rects, { a: '#8b5cf6' });
    const doc = { layers } as unknown as CanvasDocumentContractV2;
    for (const view of [fromTRS({ x: -120, y: -80 }, 0, 3, 3), fromTRS({ x: 40, y: 30 }, 0, 0.6, 0.6)]) {
      const steps = [() => repaint(caches, 'a', { height: 30, width: 30, x: 50, y: 45 }, '#fb7185')];
      const { clipped, full } = run(doc, caches, backend, view, steps, true);
      expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
    }
  });

  it('repaints in full when the damage names a layer that is gone', () => {
    // A stale id cannot be placed, so the frame must widen rather than guess.
    const layers = [rasterLayer('a')];
    const rects = { a: { height: 200, width: 300, x: 40, y: 30 } };
    const { backend, caches } = scene(layers, rects, { a: '#3b82f6' });
    const doc = { layers } as unknown as CanvasDocumentContractV2;
    const steps = [
      () => {
        repaint(caches, 'a', { height: 80, width: 90, x: 60, y: 60 }, '#ef4444');
        return { layerId: 'nonexistent', rect: { height: 1, width: 1, x: 0, y: 0 } };
      },
    ];
    const { clipped, full } = run(doc, caches, backend, IDENTITY_VIEW, steps, true);
    expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
  });

  it('holds with the checkerboard off', () => {
    const layers = [rasterLayer('a')];
    const rects = { a: { height: 200, width: 300, x: 40, y: 30 } };
    const { backend, caches } = scene(layers, rects, { a: '#3b82f6' });
    const doc = { layers } as unknown as CanvasDocumentContractV2;
    const steps = [() => repaint(caches, 'a', { height: 50, width: 50, x: 70, y: 70 }, '#ef4444')];
    const { clipped, full } = run(doc, caches, backend, IDENTITY_VIEW, steps, false);
    expect(countDifferences(full, clipped)).toEqual({ differing: 0, worst: 0 });
  });
});

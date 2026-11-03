/**
 * Real-canvas tests require damage-only adjusted-surface refreshes to match full rebuilds pixel for pixel, and
 * spatial stacks to settle on the worker's exact result.
 */

import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { applyAdjustments, perPixelAdjustments, toSpatialPlan } from '@workbench/canvas-engine/render/adjustments';
import { createDerivedSurfaceCache } from '@workbench/canvas-engine/render/derivedSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { evaluateSpatialSteps } from '@workbench/canvas-engine/render/spatial/evaluateSpatialSteps';
import {
  createSpatialAdjustmentService,
  type SpatialAdjustmentService,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { afterEach, describe, expect, it } from 'vitest';

const WIDTH = 320;
const HEIGHT = 240;
// Noise is position dependent: partial refreshes must place it by layer-local pixel, as a full rebuild does.
const ADJUSTMENTS: CanvasAdjustmentsContract = [
  { brightness: 0.25, contrast: 0.15, id: 'bc', isEnabled: true, type: 'brightness-contrast' },
  {
    amount: 0.2,
    distribution: 'gaussian',
    id: 'grain',
    isEnabled: true,
    grain: 'color',
    response: 'flat',
    size: 1,
    seed: 12345,
    type: 'noise',
  },
  { id: 'sat', isEnabled: true, saturation: -0.2, type: 'hsl' },
];

/** A layer cache seeded with a gradient, plus an adjusted cache over it. */
const setup = (withDamageTracking: boolean) => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  layers.getOrCreate('L', WIDTH, HEIGHT);
  const entry = layers.get('L')!;
  const gradient = entry.surface.ctx.createLinearGradient(0, 0, WIDTH, HEIGHT);
  gradient.addColorStop(0, '#1d4ed8');
  gradient.addColorStop(1, '#f59e0b');
  entry.surface.ctx.fillStyle = gradient;
  entry.surface.ctx.fillRect(0, 0, WIDTH, HEIGHT);
  layers.publishPixels('L');
  const adjusted = createAdjustedSurfaceCache(backend, createDerivedSurfaceCache(), {
    damageSince: withDamageTracking ? (layerId, version) => layers.damageSince(layerId, version) : undefined,
  });
  return { adjusted, entry, layers };
};

/** Paints a rect of the layer cache and publishes it with (or without) damage. */
const paint = (
  layers: ReturnType<typeof createLayerCacheStore>,
  rect: Rect,
  color: string,
  reportDamage: boolean
): void => {
  const entry = layers.get('L')!;
  const ctx = entry.surface.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = color;
  ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
  layers.publishPixels('L', reportDamage ? rect : null);
};

const pixelsOf = (surface: { ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D }) =>
  surface.ctx.getImageData(0, 0, WIDTH, HEIGHT).data;

const worstDifference = (a: Uint8ClampedArray, b: Uint8ClampedArray): number => {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  }
  return worst;
};

const compare = (steps: { rect: Rect; color: string }[], reportDamage = true) => {
  const partial = setup(true);
  const wholesale = setup(false);
  let partialPixels = new Uint8ClampedArray();
  let wholesalePixels = new Uint8ClampedArray();
  for (const { color, rect } of steps) {
    paint(partial.layers, rect, color, reportDamage);
    paint(wholesale.layers, rect, color, reportDamage);
    partialPixels = pixelsOf(partial.adjusted.get('L', partial.layers.get('L')!, ADJUSTMENTS)!);
    wholesalePixels = pixelsOf(wholesale.adjusted.get('L', wholesale.layers.get('L')!, ADJUSTMENTS)!);
  }
  return { partialPixels, wholesalePixels };
};

describe('partially-refreshed adjusted surfaces match a full rebuild', () => {
  it('tracks a single painted band', () => {
    const { partialPixels, wholesalePixels } = compare([
      { color: '#22c55e', rect: { height: 60, width: 80, x: 40, y: 30 } },
    ]);
    expect(worstDifference(partialPixels, wholesalePixels)).toBe(0);
  });

  it('tracks many successive bands across the surface', () => {
    const steps = Array.from({ length: 10 }, (_, i) => ({
      color: i % 2 === 0 ? '#22c55e' : '#ef4444',
      rect: { height: 40, width: 50, x: 10 + i * 28, y: 20 + i * 18 },
    }));
    const { partialPixels, wholesalePixels } = compare(steps);
    expect(worstDifference(partialPixels, wholesalePixels)).toBe(0);
  });

  it('tracks overlapping bands', () => {
    const { partialPixels, wholesalePixels } = compare([
      { color: '#22c55e', rect: { height: 90, width: 90, x: 50, y: 50 } },
      { color: '#a855f7', rect: { height: 90, width: 90, x: 90, y: 80 } },
      { color: '#facc15', rect: { height: 40, width: 200, x: 20, y: 100 } },
    ]);
    expect(worstDifference(partialPixels, wholesalePixels)).toBe(0);
  });

  it('tracks a band touching the surface edges', () => {
    const { partialPixels, wholesalePixels } = compare([
      { color: '#22c55e', rect: { height: HEIGHT, width: 30, x: 0, y: 0 } },
      { color: '#ef4444', rect: { height: 25, width: WIDTH, x: 0, y: HEIGHT - 25 } },
    ]);
    expect(worstDifference(partialPixels, wholesalePixels)).toBe(0);
  });

  it('rebuilds in full when a later write does not report its damage', () => {
    // Reported then unreported writes must force a full refresh; skipping the unknown step leaves its pixels
    // unadjusted. Both writes are needed to distinguish partial-refresh bugs.
    const partial = setup(true);
    const wholesale = setup(false);
    const built = { height: 60, width: 80, x: 40, y: 30 };
    const reported = { height: 50, width: 60, x: 20, y: 150 };
    const unreported = { height: 70, width: 120, x: 140, y: 120 };
    for (const scene of [partial, wholesale]) {
      paint(scene.layers, built, '#22c55e', true);
      scene.adjusted.get('L', scene.layers.get('L')!, ADJUSTMENTS);
      paint(scene.layers, reported, '#0ea5e9', true);
      paint(scene.layers, unreported, '#a855f7', false);
    }
    const a = pixelsOf(partial.adjusted.get('L', partial.layers.get('L')!, ADJUSTMENTS)!);
    const b = pixelsOf(wholesale.adjusted.get('L', wholesale.layers.get('L')!, ADJUSTMENTS)!);
    expect(worstDifference(a, b)).toBe(0);
  });

  it('rebuilds in full when the surface is reallocated under it', () => {
    // Growth moves the surface origin, so every recorded band is void.
    const { adjusted, layers } = setup(true);
    paint(layers, { height: 60, width: 80, x: 40, y: 30 }, '#22c55e', true);
    adjusted.get('L', layers.get('L')!, ADJUSTMENTS);
    layers.growToRect('L', { height: HEIGHT + 80, width: WIDTH + 60, x: -30, y: -40 });
    const grown = layers.get('L')!;
    grown.surface.ctx.fillStyle = '#0ea5e9';
    grown.surface.ctx.fillRect(0, 0, 60, 40);
    layers.publishPixels('L', { height: 40, width: 60, x: 0, y: 0 });

    const reference = setup(false);
    reference.layers.growToRect('L', { height: HEIGHT + 80, width: WIDTH + 60, x: -30, y: -40 });
    const referenceEntry = reference.layers.get('L')!;
    referenceEntry.surface.ctx.drawImage(grown.surface.canvas, 0, 0);
    reference.layers.publishPixels('L');

    const after = adjusted.get('L', grown, ADJUSTMENTS)!;
    const expected = reference.adjusted.get('L', referenceEntry, ADJUSTMENTS)!;
    const size = grown.surface.width * grown.surface.height * 4;
    const a = after.ctx.getImageData(0, 0, grown.surface.width, grown.surface.height).data;
    const b = expected.ctx.getImageData(0, 0, grown.surface.width, grown.surface.height).data;
    expect(a.length).toBe(size);
    expect(worstDifference(a, b)).toBe(0);
  });

  it('keeps noise anchored to layer-local pixels when the surface origin moves', () => {
    const { adjusted, layers } = setup(true);
    const before = pixelsOf(adjusted.get('L', layers.get('L')!, ADJUSTMENTS)!);
    layers.growToRect('L', { height: HEIGHT + 80, width: WIDTH + 60, x: -30, y: -40 });
    const grown = adjusted.get('L', layers.get('L')!, ADJUSTMENTS)!;
    // The original pixels now sit 30 right and 40 down in the surface, at unchanged layer-local coordinates.
    expect(worstDifference(grown.ctx.getImageData(30, 40, WIDTH, HEIGHT).data, before)).toBe(0);
  });
});

describe('spatially adjusted surfaces', () => {
  // Per-pixel runs on both sides of the spatial step exercise the stack's order across the worker boundary.
  const STACK: CanvasAdjustmentsContract = [
    { brightness: 0.1, contrast: 0.05, id: 'bc', isEnabled: true, type: 'brightness-contrast' },
    { amount: 70, id: 'clarity', isEnabled: true, size: 24, type: 'clarity' },
    { id: 'sat', isEnabled: true, saturation: -0.2, type: 'hsl' },
  ];

  const services: SpatialAdjustmentService[] = [];
  afterEach(() => {
    for (const service of services.splice(0)) {
      service.dispose();
    }
  });

  /** A gradient layer with a transparent border, so the kernel's alpha-bounds domain is a strict sub-rectangle. */
  const setupSpatial = () => {
    const backend = createDomRasterBackend();
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('L', WIDTH, HEIGHT);
    const entry = layers.get('L')!;
    const gradient = entry.surface.ctx.createLinearGradient(0, 0, WIDTH, HEIGHT);
    gradient.addColorStop(0, '#1d4ed8');
    gradient.addColorStop(0.5, '#e2e8f0');
    gradient.addColorStop(1, '#f59e0b');
    entry.surface.ctx.fillStyle = gradient;
    entry.surface.ctx.fillRect(16, 12, WIDTH - 40, HEIGHT - 30);
    // A dark block gives the clarity step edges to work on.
    entry.surface.ctx.fillStyle = '#0f172a';
    entry.surface.ctx.fillRect(150, 40, 60, 50);
    layers.publishPixels('L');
    const budget = new RasterMemoryBudgetController({ budgetBytes: 64 * 1024 * 1024 });
    const service = createSpatialAdjustmentService({
      reserve: (bytes) => budget.reserveOperation(bytes, { purpose: 'spatial-adjustment' }),
    });
    services.push(service);
    let settled: (() => void) | null = null;
    const adjusted = createAdjustedSurfaceCache(backend, createDerivedSurfaceCache(), {
      damageSince: (layerId, version) => layers.damageSince(layerId, version),
      spatial: { onSettle: () => settled?.(), run: (request) => service.run(request) },
    });
    const nextSettle = () =>
      new Promise<void>((resolve) => {
        settled = resolve;
      });
    const show = () => adjusted.get('L', layers.get('L')!, STACK)!;
    return { adjusted, backend, layers, nextSettle, show };
  };

  /** Reads back through a canvas as the display does, so unpremultiplied values round like the shown pixels. */
  const roundTrip = (backend: ReturnType<typeof createDomRasterBackend>, pixels: Uint8ClampedArray) => {
    const scratch = backend.createSurface(WIDTH, HEIGHT);
    const image = scratch.ctx.createImageData(WIDTH, HEIGHT);
    image.data.set(pixels);
    scratch.ctx.putImageData(image, 0, 0);
    return pixelsOf(scratch);
  };

  /** The whole stack evaluated on the main thread: what the worker path must land exactly. */
  const evaluateStack = (source: Uint8ClampedArray): Uint8ClampedArray => {
    const pixels = source.slice();
    for (const run of toSpatialPlan(STACK)) {
      if (run.kind === 'per-pixel') {
        applyAdjustments({ data: pixels, height: HEIGHT, width: WIDTH }, run.adjustments, { x: 0, y: 0 });
      } else {
        expect(evaluateSpatialSteps(pixels, WIDTH, HEIGHT, run.steps)).toEqual({ status: 'ok' });
      }
    }
    return pixels;
  };

  it('shows the per-pixel interim under a stroke, then settles to exactly the evaluated stack', async () => {
    const scene = setupSpatial();
    const firstSettle = scene.nextSettle();
    const shown = scene.show();
    expect(scene.adjusted.peek('L', scene.layers.get('L')!, STACK)).toBeNull();
    await firstSettle;
    const settledPixels = pixelsOf(shown);
    const source = () => pixelsOf(scene.layers.get('L')!.surface);
    const perPixelOnly = () => {
      const pixels = source();
      applyAdjustments({ data: pixels, height: HEIGHT, width: WIDTH }, perPixelAdjustments(STACK), { x: 0, y: 0 });
      return pixels;
    };
    expect(worstDifference(settledPixels, roundTrip(scene.backend, evaluateStack(source())))).toBe(0);
    // The clarity step really changed pixels, so the comparisons below can tell interim from result.
    expect(worstDifference(settledPixels, roundTrip(scene.backend, perPixelOnly()))).toBeGreaterThan(0);
    expect(scene.adjusted.peek('L', scene.layers.get('L')!, STACK)).toBe(shown);

    const stroke = { height: 30, width: 50, x: 60, y: 70 };
    const secondSettle = scene.nextSettle();
    paint(scene.layers, stroke, '#22c55e', true);
    expect(scene.show()).toBe(shown);

    // Inside the stroke: the new pixels with the per-pixel part only. Outside: the settled result, untouched.
    const interim = pixelsOf(shown);
    const perPixel = perPixelOnly();
    const expectedInterim = settledPixels.slice();
    for (let y = stroke.y; y < stroke.y + stroke.height; y++) {
      const start = (y * WIDTH + stroke.x) * 4;
      expectedInterim.set(perPixel.subarray(start, start + stroke.width * 4), start);
    }
    expect(worstDifference(interim, roundTrip(scene.backend, expectedInterim))).toBe(0);
    expect(scene.adjusted.peek('L', scene.layers.get('L')!, STACK)).toBeNull();

    await secondSettle;
    expect(worstDifference(pixelsOf(shown), roundTrip(scene.backend, evaluateStack(source())))).toBe(0);
    expect(scene.adjusted.peek('L', scene.layers.get('L')!, STACK)).toBe(shown);
  });
});

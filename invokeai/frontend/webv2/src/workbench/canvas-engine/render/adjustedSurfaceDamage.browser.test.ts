/**
 * The correctness guard for partially-refreshed adjusted surfaces.
 *
 * A live stroke bumps its layer's cache version every tick, which defeats the
 * adjusted-surface memo — so on an adjusted raster layer the surface was being
 * rebuilt from the whole layer every frame: a full readback plus a per-pixel JS
 * pass, 35 ms on a 2600×2100 layer. It now refreshes only the band the stroke
 * reported writing.
 *
 * That is only safe if a partial refresh is indistinguishable from a rebuild, so
 * these tests drive real canvases and compare the two pixel for pixel.
 */

import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createDerivedSurfaceCache } from '@workbench/canvas-engine/render/derivedSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

const WIDTH = 320;
const HEIGHT = 240;
const ADJUSTMENTS = {
  brightness: 0.25,
  contrast: 0.15,
  hue: 0,
  saturation: -0.2,
} as unknown as CanvasAdjustmentsContract;

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
  const adjusted = createAdjustedSurfaceCache(
    backend,
    createDerivedSurfaceCache(),
    withDamageTracking ? (layerId, version) => layers.damageSince(layerId, version) : undefined
  );
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

/**
 * Runs `steps` against two independent caches — one refreshing partially, one
 * always rebuilding — and returns both results.
 */
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
    // The case that matters is MIXED: a reported write followed by an unreported
    // one. Treating the unreported write as "nothing changed there" would refresh
    // only the first band and leave the second one's pixels unadjusted, so this
    // is what pins `damageSince` bailing out rather than skipping the step.
    // Two writes have to follow the build for this to bite: one reported, one
    // not. With only the unreported one there is nothing to refresh either way,
    // so the bail-out and a "skip the unknown step" bug look identical.
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
});

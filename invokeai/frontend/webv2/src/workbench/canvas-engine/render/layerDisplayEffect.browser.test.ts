/** Real-canvas check that a floating selection's baked display copy matches the layer pixels it was lifted from. */

import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';

import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { renderLayerDisplayEffect } from '@workbench/canvas-engine/render/layerDisplayEffect';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { liftSelectedPixels } from '@workbench/canvas-engine/selection/floatingSelection';
import { describe, expect, it } from 'vitest';

describe('floating selection display effect', () => {
  it("bakes the layer's adjusted pixels at the lift rect, noise included", () => {
    const backend = createDomRasterBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.growToRect('L', { height: 30, width: 40, x: -7, y: -5 });
    for (let x = 0; x < 40; x += 1) {
      entry.surface.ctx.fillStyle = `rgb(${60 + x * 3}, 140, ${210 - x * 3})`;
      entry.surface.ctx.fillRect(x, 0, 1, 30);
    }
    layers.publishPixels('L');
    const layer = {
      adjustments: [
        { amount: 0.25, distribution: 'uniform', id: 'n', isEnabled: true, monochrome: false, seed: 77, type: 'noise' },
      ],
      blendMode: 'normal',
      id: 'L',
      isEnabled: true,
      isLocked: false,
      name: 'L',
      opacity: 1,
      source: { bitmap: { height: 30, imageName: 'l.png', width: 40 }, offset: { x: -7, y: -5 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 3, y: 2 },
      type: 'raster',
    } satisfies CanvasRasterLayerContractV2;
    const mask = backend.createSurface(12, 9);
    mask.ctx.fillStyle = '#ffffff';
    mask.ctx.fillRect(0, 0, 12, 9);

    const lifted = liftSelectedPixels({
      backend,
      cache: { rect: entry.rect, surface: entry.surface },
      layerMatrix: { a: 1, b: 0, c: 0, d: 1, e: 3, f: 2 },
      mask: { rect: { height: 9, width: 12, x: 5, y: 4 }, surface: mask },
    })!;
    const display = renderLayerDisplayEffect(backend, layer, lifted.pixels.surface, lifted.pixels.rect)!;

    // The mask's document rect lands at layer-local (2, 2), surface (9, 7).
    expect(lifted.pixels.rect).toEqual({ height: 9, width: 12, x: 2, y: 2 });
    const adjusted = createAdjustedSurfaceCache(backend).get('L', entry, layer.adjustments)!;
    expect([...display.ctx.getImageData(0, 0, 12, 9).data]).toEqual([...adjusted.ctx.getImageData(9, 7, 12, 9).data]);
  });
});

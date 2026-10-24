/** Real-canvas checks that rasterizing keeps a noisy layer's displayed grain. */

import type { CanvasDocumentContractV3, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { rasterizeSource, type RasterizeDeps } from '@workbench/canvas-engine/render/rasterizers';
import { describe, expect, it } from 'vitest';

import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { RasterizeLayerController } from './rasterizeLayerController';

const NOISE = {
  amount: 0.3,
  distribution: 'gaussian',
  grain: 'color',
  id: 'grain',
  isEnabled: true,
  response: 'flat',
  seed: 13,
  size: 1,
  type: 'noise',
} as const;
const TRANSFORM = { rotation: Math.PI / 9, scaleX: 0.6, scaleY: 1.3, x: 11, y: 5 };

const noisyLayer = (source: CanvasRasterLayerContractV2['source']): CanvasRasterLayerContractV2 => ({
  adjustments: [NOISE, { brightness: 0.1, contrast: 0.2, id: 'bc', isEnabled: true, type: 'brightness-contrast' }],
  blendMode: 'normal',
  id: 'layer',
  isEnabled: true,
  isLocked: false,
  name: 'Layer',
  opacity: 1,
  source,
  transform: TRANSFORM,
  type: 'raster',
});

describe('RasterizeLayerController with a position-dependent stack', () => {
  it.each([
    [
      'shape',
      noisyLayer({ fill: '#808080', height: 30, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 40 }),
    ],
    // Text resizes its scratch to measured bounds; this text fits its estimate, so nothing is clipped.
    [
      'text',
      noisyLayer({
        align: 'left',
        color: '#808080',
        content: 'Iii\nill',
        fontFamily: 'sans-serif',
        fontSize: 28,
        fontWeight: 700,
        lineHeight: 1.2,
        type: 'text',
      }),
    ],
  ])('bakes the displayed grain of a rotated, scaled %s layer and drops the stack until undo', async (_, layer) => {
    const backend = createDomRasterBackend();
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 64, width: 64, x: 0, y: 0 },
      height: 64,
      selectedLayerId: 'layer',
      stacks: stacksFrom([layer]),
      version: 3,
      width: 64,
    };
    const h = createLayerOperationHarness(document, { backend });
    const rasterizeDeps = (): RasterizeDeps => ({
      backend,
      documentSize: { height: 64, width: 64 },
      resolver: () => Promise.reject(new Error('parametric sources need no images')),
      store: h.layers,
    });
    const controller = new RasterizeLayerController({ backend, ctx: h.ctx, rasterizeDeps });

    // The display: the layer's rasterized cache, adjusted by the display's cache, resampled through the transform.
    const rendered = await rasterizeSource(layer.source, rasterizeDeps());
    const displayCache = createLayerCacheStore(backend);
    const entry = displayCache.publishRasterized('layer', rendered.rect, rendered.surface);
    if (layer.source.type === 'text') {
      expect(entry.rect.width).toBeLessThan(getSourceContentRect(layer, document).width);
    }
    const adjusted = createAdjustedSurfaceCache(backend).get('layer', entry, layer.adjustments)!;
    const displayed = (rect: Rect): number[] => {
      const surface = backend.createSurface(rect.width, rect.height);
      const { rotation, scaleX, scaleY, x, y } = TRANSFORM;
      const matrix = fromTRS({ x, y }, rotation, scaleX, scaleY);
      surface.ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - rect.x, matrix.f - rect.y);
      surface.ctx.drawImage(adjusted.canvas, entry.rect.x, entry.rect.y);
      return [...surface.ctx.getImageData(0, 0, rect.width, rect.height).data];
    };

    expect(controller.rasterize('layer')).toBe('rasterized');

    const paint = getDocumentLayer(h.document(), 'layer');
    expect(paint).toMatchObject({ source: { type: 'paint' }, transform: { rotation: 0, scaleX: 1, scaleY: 1 } });
    expect(paint).not.toHaveProperty('adjustments');
    const baked = h.layers.get('layer')!;
    const bakedPixels = [...baked.surface.ctx.getImageData(0, 0, baked.rect.width, baked.rect.height).data];
    expect(bakedPixels).toEqual(displayed(baked.rect));

    await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLayer(h.document(), 'layer')).toEqual(layer);
  });
});

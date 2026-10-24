/** Real-canvas checks that world-space bakes resample the display's layer-local adjusted pixels. */

import type { CanvasDocumentContractV3, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createSpatialAdjustmentService } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { describe, expect, it, vi } from 'vitest';

import { RasterExportController } from './rasterExportController';

const TRANSFORM = { rotation: Math.PI / 9, scaleX: 0.6, scaleY: 0.6, x: 11, y: 5 };

/** A rotated, downscaled noisy layer with off-origin content: its stack works in layer-local pixels. */
const bakeScene = (peekDisplayed: boolean) => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  const entry = layers.growToRect('layer', { height: 30, width: 40, x: -6, y: -4 });
  for (let x = 0; x < 40; x += 1) {
    entry.surface.ctx.fillStyle = `rgb(${60 + x * 3}, 128, ${200 - x * 3})`;
    entry.surface.ctx.fillRect(x, 0, 1, 30);
  }
  layers.publishPixels('layer');
  const layer: CanvasRasterLayerContractV2 = {
    adjustments: [
      {
        amount: 0.2,
        distribution: 'gaussian',
        id: 'n',
        isEnabled: true,
        grain: 'color',
        response: 'flat',
        size: 1,
        seed: 9,
        type: 'noise',
      },
      { id: 'i', isEnabled: true, type: 'invert' },
    ],
    blendMode: 'normal',
    id: 'layer',
    isEnabled: true,
    isLocked: false,
    name: 'Layer',
    opacity: 1,
    source: { bitmap: { height: 30, imageName: 'layer.png', width: 40 }, offset: { x: -6, y: -4 }, type: 'paint' },
    transform: TRANSFORM,
    type: 'raster',
  };
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 64, width: 64, x: 0, y: 0 },
    height: 64,
    stacks: stacksFrom([layer]),
    selectedLayerId: null,
    version: 3,
    width: 64,
  };
  const displayed = createAdjustedSurfaceCache(backend);
  const reserve = vi.fn((_bytes: number) => ({ lease: { release: () => undefined }, status: 'ok' as const }));
  const controller = new RasterExportController({
    backend,
    captureGuard: (captured) => ({
      cacheVersion: 1,
      documentGeneration: 1,
      layer: captured,
      layerId: captured.id,
      projectId: 'p',
    }),
    getDocument: () => document,
    getOrStartRasterization: () => Promise.resolve('published'),
    isGuardCurrent: () => true,
    isRasterizing: () => false,
    isSupportedSource: () => true,
    layers,
    ...(peekDisplayed
      ? {
          peekAdjustedCacheSurface: (layerId: string, surface: RasterSurface) =>
            surface === layers.get(layerId)!.surface
              ? displayed.get(layerId, layers.get(layerId)!, layer.adjustments)
              : null,
        }
      : {}),
    pin: () => ({ release: () => undefined }),
    reserve,
  });
  /** The display's adjusted surface drawn through the layer transform: what the canvas shows. */
  const reference = (rect: { x: number; y: number; width: number; height: number }) => {
    const surface = backend.createSurface(rect.width, rect.height);
    const matrix = fromTRS({ x: TRANSFORM.x, y: TRANSFORM.y }, TRANSFORM.rotation, TRANSFORM.scaleX, TRANSFORM.scaleY);
    surface.ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - rect.x, matrix.f - rect.y);
    surface.ctx.drawImage(displayed.get('layer', layers.get('layer')!, layer.adjustments)!.canvas, -6, -4);
    return [...surface.ctx.getImageData(0, 0, rect.width, rect.height).data];
  };
  return { controller, reference, reserve };
};

describe('baked raster export', () => {
  it.each([
    ['adjusting a reserved layer-local copy', false],
    ["drawing the display's current adjusted copy", true],
  ])("keeps a rotated, downscaled noisy layer's displayed grain by %s", async (_, peekDisplayed) => {
    const { controller, reference, reserve } = bakeScene(peekDisplayed);

    const baked = await controller.baked('layer');

    expect(baked.status).toBe('ok');
    if (baked.status !== 'ok') {
      return;
    }
    const { height, width } = baked.rect;
    expect([...baked.surface.ctx.getImageData(0, 0, width, height).data]).toEqual(reference(baked.rect));
    // The world surface always; a layer-local copy (surface plus ImageData) only without a current display copy.
    const world = width * height * 4;
    expect(reserve.mock.calls).toEqual(peekDisplayed ? [[world]] : [[40 * 30 * 8], [world]]);
    baked.release();
  });
});

describe('baked raster export with a spatial stack', () => {
  it("bakes a rotated, downscaled layer's settled display, evaluated in the spatial worker", async () => {
    const backend = createDomRasterBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.growToRect('layer', { height: 30, width: 40, x: -6, y: -4 });
    for (let x = 0; x < 40; x += 1) {
      entry.surface.ctx.fillStyle = `rgb(${60 + x * 3}, ${x % 5 === 0 ? 20 : 128}, ${200 - x * 3})`;
      entry.surface.ctx.fillRect(x, 2, 1, 26);
    }
    layers.publishPixels('layer');
    const layer: CanvasRasterLayerContractV2 = {
      adjustments: [
        { amount: 90, id: 'clarity', isEnabled: true, size: 8, type: 'clarity' },
        { amount: 50, id: 'dehaze', isEnabled: true, type: 'dehaze' },
      ],
      blendMode: 'normal',
      id: 'layer',
      isEnabled: true,
      isLocked: false,
      name: 'Layer',
      opacity: 1,
      source: { bitmap: { height: 30, imageName: 'layer.png', width: 40 }, offset: { x: -6, y: -4 }, type: 'paint' },
      transform: TRANSFORM,
      type: 'raster',
    };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 64, width: 64, x: 0, y: 0 },
      height: 64,
      stacks: stacksFrom([layer]),
      selectedLayerId: null,
      version: 3,
      width: 64,
    };
    const memory = new RasterMemoryBudgetController({ budgetBytes: 64 * 1024 * 1024 });
    const service = createSpatialAdjustmentService({
      reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'spatial-adjustment' }),
    });
    try {
      const display = createAdjustedSurfaceCache(backend, undefined, {
        spatial: { run: (request) => service.run(request) },
      });
      await vi.waitFor(
        () => {
          display.get('layer', entry, layer.adjustments);
          expect(display.peek('layer', entry, layer.adjustments)).not.toBeNull();
        },
        { interval: 20, timeout: 10_000 }
      );
      const controller = new RasterExportController({
        backend,
        captureGuard: (captured) => ({
          cacheVersion: 1,
          documentGeneration: 1,
          layer: captured,
          layerId: captured.id,
          projectId: 'p',
        }),
        getDocument: () => document,
        getOrStartRasterization: () => Promise.resolve('published'),
        isGuardCurrent: () => true,
        isRasterizing: () => false,
        isSupportedSource: () => true,
        layers,
        pin: () => ({ release: () => undefined }),
        runSpatial: (request) => service.run(request),
      });

      const baked = await controller.baked('layer');

      expect(baked.status).toBe('ok');
      if (baked.status !== 'ok') {
        return;
      }
      const { height, width } = baked.rect;
      const reference = backend.createSurface(width, height);
      const matrix = fromTRS(
        { x: TRANSFORM.x, y: TRANSFORM.y },
        TRANSFORM.rotation,
        TRANSFORM.scaleX,
        TRANSFORM.scaleY
      );
      reference.ctx.setTransform(
        matrix.a,
        matrix.b,
        matrix.c,
        matrix.d,
        matrix.e - baked.rect.x,
        matrix.f - baked.rect.y
      );
      reference.ctx.drawImage(display.peek('layer', entry, layer.adjustments)!.canvas, -6, -4);
      expect([...baked.surface.ctx.getImageData(0, 0, width, height).data]).toEqual([
        ...reference.ctx.getImageData(0, 0, width, height).data,
      ]);
      baked.release();
    } finally {
      service.dispose();
    }
  });
});

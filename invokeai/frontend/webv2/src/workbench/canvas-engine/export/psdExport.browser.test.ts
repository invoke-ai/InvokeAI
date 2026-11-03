/** Real-worker check that a PSD leaf carries what the settled display shows for its own spatial stack. */

import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';
import type { Psd } from 'ag-psd';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { executePsdExport, planPsdExport } from '@workbench/canvas-engine/export/psdExport';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createSpatialAdjustmentService } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { describe, expect, it, vi } from 'vitest';

const CONTENT: Rect = { height: 30, width: 40, x: -6, y: -4 };
const TRANSFORM = { rotation: Math.PI / 9, scaleX: 0.6, scaleY: 0.8, x: 11, y: 5 };
const STACK: CanvasAdjustmentsContract = [
  { amount: 100, id: 'texture', isEnabled: true, size: 3, type: 'texture' },
  { id: 'invert', isEnabled: true, type: 'invert' },
  { amount: 70, id: 'clarity', isEnabled: true, size: 10, type: 'clarity' },
];

describe('executePsdExport with a spatial stack', () => {
  it("bakes a rotated, scaled leaf's settled display, layer-local before the transform", async () => {
    const backend = createDomRasterBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.growToRect('layer', CONTENT);
    for (let x = 0; x < CONTENT.width; x += 1) {
      entry.surface.ctx.fillStyle = `rgb(${40 + x * 5}, ${x % 7 === 0 ? 30 : 140}, ${220 - x * 4})`;
      entry.surface.ctx.fillRect(x, 3, 1, CONTENT.height - 6);
    }
    layers.publishPixels('layer');
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
          display.get('layer', entry, STACK);
          expect(display.peek('layer', entry, STACK)).not.toBeNull();
        },
        { interval: 20, timeout: 10_000 }
      );
      const plan = planPsdExport([
        {
          adjustments: STACK,
          blendMode: 'normal',
          contentRect: CONTENT,
          id: 'layer',
          isEnabled: true,
          name: 'Layer',
          opacity: 1,
          transform: TRANSFORM,
        },
      ]);
      expect(plan.status).toBe('ok');
      let written: Psd | null = null;

      await executePsdExport(plan, 'layer.psd', {
        backend,
        download: () => undefined,
        getLayerSurface: () => Promise.resolve({ rect: entry.rect, surface: entry.surface }),
        runSpatial: (request) => service.run(request),
        writePsd: (psd) => {
          written = psd;
          return Promise.resolve(new ArrayBuffer(0));
        },
      });

      // What the canvas shows: the display's exact adjusted surface resampled through the transform.
      const worldRect = plan.status === 'ok' ? plan.layers[0]!.worldRect : CONTENT;
      const reference = backend.createSurface(worldRect.width, worldRect.height);
      const { rotation, scaleX, scaleY, x, y } = TRANSFORM;
      const matrix = fromTRS({ x, y }, rotation, scaleX, scaleY);
      reference.ctx.setTransform(
        matrix.a,
        matrix.b,
        matrix.c,
        matrix.d,
        matrix.e - worldRect.x,
        matrix.f - worldRect.y
      );
      reference.ctx.drawImage(display.peek('layer', entry, STACK)!.canvas, CONTENT.x, CONTENT.y);
      const expected = reference.ctx.getImageData(0, 0, worldRect.width, worldRect.height).data;
      expect([...written!.children![0]!.imageData!.data]).toEqual([...expected]);
      expect(memory.snapshot().reservedBytes).toBe(0);
    } finally {
      service.dispose();
    }
  });
});

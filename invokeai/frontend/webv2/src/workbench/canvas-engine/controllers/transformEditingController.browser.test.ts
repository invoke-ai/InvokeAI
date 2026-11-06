import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { TransformSession } from '@workbench/canvas-engine/engineStores';
import type { Rect } from '@workbench/canvas-engine/types';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createDomRasterBackend, type RasterSurface } from '@workbench/canvas-engine/render/raster';
import { createSpatialAdjustmentService } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { describe, expect, it, vi } from 'vitest';

import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { TransformEditingController } from './transformEditingController';

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
const COLOR = { brightness: 0.1, contrast: 0.2, id: 'bc', isEnabled: true, type: 'brightness-contrast' } as const;
const TRANSFORM = { rotation: Math.PI / 9, scaleX: 0.6, scaleY: 1.3, x: 11, y: 5 };
const RECT = { x: -3, y: 4, width: 32, height: 24 };

describe('Transform adjustment bakes', () => {
  it.each([
    ['noise', true],
    ['noise', false],
    ['spatial', true],
    ['spatial', false],
  ] as const)(
    'bakes settled %s pixels before dropping the stack (smoothing %s), with raw undo',
    async (kind, smoothing) => {
      const backend = createDomRasterBackend();
      const memory = new RasterMemoryBudgetController({ budgetBytes: 64 * 1024 * 1024 });
      const service = createSpatialAdjustmentService({
        reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'spatial-adjustment' }),
      });
      const layer = {
        ...layerContract('layer'),
        adjustments:
          kind === 'noise'
            ? [NOISE, COLOR]
            : [
                { amount: 120, id: 'clarity', isEnabled: true, size: 6, type: 'clarity' },
                COLOR,
                { amount: 80, id: 'dehaze', isEnabled: true, type: 'dehaze' },
              ],
        source: {
          type: 'paint',
          bitmap: { imageName: 'source.png', width: RECT.width, height: RECT.height },
          offset: { x: RECT.x, y: RECT.y },
        },
      } as CanvasRasterLayerContractV2;
      const h = createLayerOperationHarness(documentFrom([layer], layer.id), { backend });
      let session: TransformSession | null = null;
      const source = backend.createSurface(RECT.width, RECT.height);
      const raw = source.ctx.createImageData(RECT.width, RECT.height);
      for (let y = 0; y < RECT.height; y++) {
        for (let x = 0; x < RECT.width; x++) {
          const offset = (y * RECT.width + x) * 4;
          raw.data.set([(x * 41 + y * 13) % 256, (x * 19 + y * 43) % 256, (x * 7 + y * 29) % 256, 255], offset);
        }
      }
      source.ctx.putImageData(raw, 0, 0);
      const entry = h.layers.publishRasterized(layer.id, RECT, source);
      const display = createAdjustedSurfaceCache(backend, undefined, {
        spatial: { run: (request) => service.run(request) },
      });
      const refusal = vi.fn();
      try {
        await vi.waitFor(
          () => {
            display.get(layer.id, entry, layer.adjustments);
            expect(display.peek(layer.id, entry, layer.adjustments)).not.toBeNull();
          },
          { timeout: 10_000, interval: 20 }
        );
        const adjusted = display.peek(layer.id, entry, layer.adjustments)!;
        expect([...adjusted.ctx.getImageData(0, 0, RECT.width, RECT.height).data]).not.toEqual([...raw.data]);
        const expectedTransformed = (rect: Rect, ratio: number) => {
          const target = backend.createSurface(rect.width * ratio, rect.height * ratio);
          const matrix = fromTRS(
            { x: TRANSFORM.x, y: TRANSFORM.y },
            TRANSFORM.rotation,
            TRANSFORM.scaleX,
            TRANSFORM.scaleY
          );
          target.ctx.setTransform(
            matrix.a * ratio,
            matrix.b * ratio,
            matrix.c * ratio,
            matrix.d * ratio,
            (matrix.e - rect.x) * ratio,
            (matrix.f - rect.y) * ratio
          );
          target.ctx.drawImage(adjusted.canvas, RECT.x, RECT.y);
          return target;
        };
        let expectedPixels: ImageData | null = null;
        const resample = vi.fn((surface: RasterSurface, rect: { width: number; height: number }) => {
          const worldRect = {
            ...rect,
            x: Math.floor(11 - 3 * Math.cos(Math.PI / 9) * 0.6 - 28 * Math.sin(Math.PI / 9) * 1.3),
            y: Math.floor(5 - 3 * Math.sin(Math.PI / 9) * 0.6 + 4 * Math.cos(Math.PI / 9) * 1.3),
          };
          const reference = expectedTransformed(worldRect, 2);
          expect([...surface.ctx.getImageData(0, 0, surface.width, surface.height).data]).toEqual([
            ...reference.ctx.getImageData(0, 0, reference.width, reference.height).data,
          ]);
          const output = backend.createSurface(rect.width, rect.height);
          output.ctx.drawImage(reference.canvas, 0, 0, rect.width, rect.height);
          expectedPixels = output.ctx.getImageData(0, 0, rect.width, rect.height);
          return Promise.resolve(expectedPixels);
        });
        const controller = new TransformEditingController({
          backend,
          ctx: h.ctx,
          canEdit: () => true,
          commitStructural: vi.fn(),
          enterTool: vi.fn(),
          getCache: (id) => h.layers.get(id) ?? null,
          getDocument: h.document,
          getOptions: () => ({ smoothing, resampleMode: 'lanczos', fitMode: 'contain' }),
          getUserToolId: () => 'transform',
          hasFloatingSelection: () => false,
          invalidate: vi.fn(),
          isGestureActive: () => false,
          peekCacheRect: (id) => h.layers.peek(id)?.rect,
          pending: { get: () => null, set: vi.fn() },
          rebaseLineEnd: vi.fn(),
          reportRefusal: refusal,
          resample,
          runSpatial: (request) => service.run(request),
          select: vi.fn(),
          session: {
            get: () => session,
            set: (next) => {
              session = next;
            },
          },
          setOverride: vi.fn(),
        });
        controller.begin(layer.id);
        controller.update(TRANSFORM);
        controller.apply();
        await vi.waitFor(() => expect(h.history.canUndo()).toBe(true), { timeout: 10_000, interval: 20 });
        expect(refusal).not.toHaveBeenCalled();
        expect(resample).toHaveBeenCalledTimes(smoothing ? 1 : 0);
        expect(getDocumentLayer(h.document(), layer.id)).not.toHaveProperty('adjustments');
        const baked = h.layers.get(layer.id)!;
        if (!smoothing) {
          expectedPixels = expectedTransformed(baked.rect, 1).ctx.getImageData(
            0,
            0,
            baked.rect.width,
            baked.rect.height
          );
        }
        expect([...baked.surface.ctx.getImageData(0, 0, baked.rect.width, baked.rect.height).data]).toEqual([
          ...expectedPixels!.data,
        ]);
        expect(await h.history.undo()).toEqual({ status: 'applied' });
        expect(getDocumentLayer(h.document(), layer.id)).toEqual(layer);
        expect([...h.layers.get(layer.id)!.surface.ctx.getImageData(0, 0, RECT.width, RECT.height).data]).toEqual([
          ...raw.data,
        ]);
        expect(await h.history.redo()).toEqual({ status: 'applied' });
        expect(getDocumentLayer(h.document(), layer.id)).not.toHaveProperty('adjustments');
        controller.dispose();
      } finally {
        display.dispose();
        service.dispose();
      }
    }
  );
});

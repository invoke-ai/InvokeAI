/** Real-worker check that a PSD leaf carries what the settled display shows for its own spatial stack. */

import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { executePsdExport, planPsdExport } from '@workbench/canvas-engine/export/psdExport';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createSpatialAdjustmentService } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { readPsd } from 'ag-psd';
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
      let bytes: ArrayBuffer | null = null;

      await executePsdExport(plan, 'layer.psd', {
        backend,
        download: (data) => {
          bytes = data;
        },
        getLayerSurface: () => Promise.resolve({ rect: entry.rect, surface: entry.surface }),
        runSpatial: (request) => service.run(request),
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
      const written = readPsd(bytes!, { useImageData: true });
      expect([...written.children![0]!.imageData!.data]).toEqual([...expected]);
      expect(memory.snapshot().reservedBytes).toBe(0);
    } finally {
      service.dispose();
    }
  });
});

it('round-trips nested mixed visibility and a transformed leaf through the serialization worker', async () => {
  const backend = createDomRasterBackend();
  const rect = { x: 0, y: 0, width: 2, height: 2 };
  const fill = (color: string, width = 2, height = 2) => {
    const surface = backend.createSurface(width, height);
    surface.ctx.fillStyle = color;
    surface.ctx.fillRect(0, 0, width, height);
    return surface;
  };
  const sources = { red: fill('#ff0000'), blue: fill('#0000ff'), base: fill('#00ff00', 8, 4) };
  const leaf = (id: string) => ({
    id,
    name: id,
    contentRect: rect,
    blendMode: 'normal' as const,
    isEnabled: true,
    opacity: 1,
    transform: { x: 0, y: 0, rotation: 0, scaleX: 1, scaleY: 1 },
  });
  const plan = planPsdExport([
    {
      type: 'group',
      id: 'outer',
      name: 'outer',
      isEnabled: true,
      opacity: 0.5,
      children: [
        { type: 'group', id: 'hidden', name: 'hidden', isEnabled: false, children: [leaf('blue')] },
        { ...leaf('red'), opacity: 0.5, transform: { x: 3, y: 1, rotation: 0, scaleX: 2, scaleY: 1 } },
      ],
    },
    { ...leaf('base'), contentRect: { x: 0, y: 0, width: 8, height: 4 } },
  ]);
  let bytes: ArrayBuffer | undefined;
  await executePsdExport(plan, 'nested.psd', {
    backend,
    getLayerSurface: (id) =>
      Promise.resolve({
        rect: id === 'base' ? { ...rect, width: 8, height: 4 } : rect,
        surface: sources[id as keyof typeof sources],
      }),
    download: (data) => {
      bytes = data;
    },
  });
  const psd = readPsd(bytes!, { useImageData: true });
  expect([psd.width, psd.height]).toEqual([8, 4]);
  expect(psd.children!.map((node) => node.name)).toEqual(['base', 'outer']);
  const outer = psd.children![1]!;
  expect(outer.hidden).toBe(false);
  expect(outer.opacity).toBeCloseTo(0.5, 2);
  expect(outer.children![0]!.opacity).toBeCloseTo(0.5, 2);
  expect(outer.children!.map((node) => node.name)).toEqual(['red', 'hidden']);
  expect(outer.children![0]).toMatchObject({ left: 3, top: 1, right: 7, bottom: 3, hidden: false });
  const hidden = outer.children![1]!;
  expect(hidden.hidden).toBe(true);
  expect(hidden.children![0]).toMatchObject({ name: 'blue', hidden: false });
  expect(Array.from(hidden.children![0]!.imageData!.data.subarray(0, 4))).toEqual([0, 0, 255, 255]);
  const expected: number[] = [];
  for (let y = 0; y < 4; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      expected.push(...(x >= 3 && x < 7 && y >= 1 && y < 3 ? [64, 191, 0, 255] : [0, 255, 0, 255]));
    }
  }
  expect(Array.from(psd.imageData!.data)).toEqual(expected);
});

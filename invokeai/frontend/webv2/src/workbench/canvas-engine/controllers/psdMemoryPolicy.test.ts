import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { planPsdExport } from '@workbench/canvas-engine/export/psdExport';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { writePsd, type Psd } from 'ag-psd';
import { writeDataRLE } from 'ag-psd/dist/helpers';
import { describe, expect, it, vi } from 'vitest';

import {
  getPsdRequiredBytes,
  getPsdRleBound,
  getPsdSerializationBudget,
  PSD_SERIALIZATION_ALLOWANCE_BYTES,
  PsdExportController,
} from './psdExportController';
import { RasterMemoryBudgetController } from './rasterMemoryBudgetController';

const planFor = (width: number, height: number, count = 1) => {
  const plan = planPsdExport(
    Array.from({ length: count }, (_, i) => ({
      id: String(i),
      name: String(i),
      contentRect: { x: 0, y: 0, width, height },
      opacity: 1,
      blendMode: 'normal' as const,
      isEnabled: true,
      transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0 },
    }))
  );
  if (plan.status !== 'ok') {
    throw new Error('Invalid test plan');
  }
  return plan;
};

const pixelRecord = (row: readonly number[], height = 1) => ({
  width: row.length,
  height,
  data: Uint8ClampedArray.from({ length: row.length * height * 4 }, (_, i) => row[Math.floor(i / 4) % row.length]!),
});

describe('installed ag-psd memory policy', () => {
  it('pins independent phase accounting, including all writer growth buffers and exact output slice', () => {
    const plan = planFor(2, 2);
    expect(getPsdRequiredBytes(plan)).toBe(64);
    expect(getPsdSerializationBudget(plan)).toEqual({ outputBound: 5222, requiredBytes: 26290 });
  });

  it('bounds PackBits across short runs, literal packet boundaries and adversarial transitions', () => {
    const check = (row: number[]) => {
      const pixels = pixelRecord(row);
      const encoded = writeDataRLE(new Uint8Array(8 * row.length + 8), pixels, [0, 1, 2, 3], false)!;
      expect(encoded.byteLength).toBeLessThanOrEqual(getPsdRleBound(row.length, 1));
    };
    for (let width = 1; width <= 12; width++) {
      for (let bits = 0; bits < 2 ** width; bits++) {
        check(Array.from({ length: width }, (_, i) => (bits >> i) & 1));
      }
    }
    for (const width of [127, 128, 129, 130, 131, 255, 256, 257, 2048, 4096, 30000]) {
      check(Array.from({ length: width }, (_, i) => i % 251));
      check(Array.from({ length: width }, (_, i) => Math.floor(i / 3) % 2));
      check(Array.from({ length: width }, (_, i) => (i % 131 < 128 ? i % 251 : 255)));
    }
  });

  it.each([1, 2, 127, 128, 129, 259, 2048])('bounds actual writePsd output for width %i', (width) => {
    const height = 17;
    const plan = planFor(width, height, 3);
    const imageData = pixelRecord(
      Array.from({ length: width }, (_, i) => i % 251),
      height
    );
    const psd: Psd = {
      width,
      height,
      imageData,
      children: plan.layers.map((layer) => ({
        name: layer.name,
        imageData,
        left: 0,
        top: 0,
        right: width,
        bottom: height,
      })),
    };
    expect(writePsd(psd).byteLength).toBeLessThanOrEqual(getPsdSerializationBudget(plan).outputBound);
  });

  it.each([
    [2048, 2048, 1],
    [4096, 4096, 1],
    [3840, 2160, 1],
    [1920, 1080, 3],
    [1536, 1536, 3],
  ])(
    'admits %i×%i with %i layers, live caches and detached snapshots under the unchanged raster ceiling',
    async (width, height, count) => {
      const backend = createTestStubRasterBackend();
      const memory = new RasterMemoryBudgetController();
      const liveBytes = width * height * count * 4;
      memory.setCategoryBytes('base', liveBytes);
      memory.setCategoryBytes('decoded', liveBytes);
      const snapshot = memory.trackDetached(liveBytes);
      const plan = planFor(width, height, count);
      expect(getPsdSerializationBudget(plan).requiredBytes).toBeLessThan(PSD_SERIALIZATION_ALLOWANCE_BYTES);
      const canvas = {
        ...createInitialWorkbenchState().projects[0]!.canvas,
        document: documentFrom(Array.from({ length: count }, (_, i) => layerContract(String(i)))),
      };
      const execute = vi.fn((_plan, _name, deps) => {
        expect(memory.snapshot().totalBytes).toBeLessThanOrEqual(512 * 1024 * 1024);
        expect(memory.snapshot().reservedBytes).toBe(8 * width * height * (count + 1));
        deps.onStage('serializing');
        expect(memory.snapshot()).toMatchObject({ reservedBytes: 0, detachedBytes: 0, baseBytes: liveBytes });
        return Promise.resolve();
      });
      const controller = new PsdExportController({
        backend,
        captureDocumentSnapshot: () => ({ canvas, documentGeneration: 1 }),
        captureRasterSnapshot: () =>
          Promise.resolve({
            status: 'ok' as const,
            snapshot: {
              canvas,
              documentGeneration: 1,
              emptyLayerIds: new Set<string>(),
              release: snapshot.release,
              layerSurfaces: new Map(
                plan.layers.map((layer) => [
                  layer.id,
                  { rect: layer.contentRect, surface: backend.createSurface(width, height) },
                ])
              ),
            },
          }),
        getAvailableBytes: () => memory.getAvailableBytes(),
        isDocumentSnapshotCurrent: () => true,
        reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'psd-export' }),
        execute,
      });
      await expect(controller.export('representative')).resolves.toBe('exported');
      expect(execute).toHaveBeenCalledOnce();
      controller.dispose();
      memory.dispose();
    }
  );

  it.each([
    [4429, 1, 1, 1_342_003_772, 'exported'],
    [4430, 1, 1, 1_342_608_010, 'too-large'],
    [4608, 1, 1, 1_452_550_174, 'too-large'],
    [64, 2, 64, 1_620_973_628, 'too-large'],
    [8192, 1, 1, 4_589_790_238, 'too-large'],
  ] as const)(
    'enforces the serialization ceiling for %i² content, %i layers at scale %i before executor allocation',
    async (size, count, scale, expectedBytes, result) => {
      const backend = createTestStubRasterBackend();
      const memory = new RasterMemoryBudgetController();
      const snapshot = memory.trackDetached(size * size * count * 4);
      const canvas = {
        ...createInitialWorkbenchState().projects[0]!.canvas,
        document: documentFrom(
          Array.from({ length: count }, (_, i) =>
            layerContract(String(i), 'raster', {
              name: String(i),
              transform: { x: 0, y: 0, scaleX: scale, scaleY: scale, rotation: 0 },
            })
          )
        ),
      };
      // These admission fixtures never read pixels; even the 8192² surface holds only dimensions and call logs.
      const surfaces = new Map(
        Array.from({ length: count }, (_, i) => [
          String(i),
          { rect: { x: 0, y: 0, width: size, height: size }, surface: backend.createSurface(size, size) },
        ])
      );
      const execute = vi.fn(() => Promise.resolve());
      const reserve = vi.fn((bytes: number) => memory.reserveOperation(bytes, { purpose: 'psd-export' }));
      const release = vi.fn(snapshot.release);
      const controller = new PsdExportController({
        backend,
        captureDocumentSnapshot: () => ({ canvas, documentGeneration: 1 }),
        captureRasterSnapshot: () =>
          Promise.resolve({
            status: 'ok' as const,
            snapshot: {
              canvas,
              documentGeneration: 1,
              emptyLayerIds: new Set<string>(),
              release,
              layerSurfaces: surfaces,
            },
          }),
        execute,
        reserve,
        getAvailableBytes: () => memory.getAvailableBytes(),
        isDocumentSnapshotCurrent: () => true,
      });
      expect(getPsdSerializationBudget(planFor(size * scale, size * scale, count)).requiredBytes).toBe(expectedBytes);
      await expect(controller.export('serialization-limit')).resolves.toBe(result);
      expect(execute).toHaveBeenCalledTimes(result === 'exported' ? 1 : 0);
      expect(reserve).toHaveBeenCalledTimes(result === 'exported' ? 1 : 0);
      expect(release).toHaveBeenCalledOnce();
      expect(memory.snapshot()).toMatchObject({ reservedBytes: 0, detachedBytes: 0 });
      controller.dispose();
      memory.dispose();
    }
  );
});

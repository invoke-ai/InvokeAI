import type { SpatialAdjustmentStep } from '@workbench/canvas-engine/render/spatial/spatialSteps';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { evaluateSpatialSteps } from '@workbench/canvas-engine/render/spatial/evaluateSpatialSteps';
import {
  createSpatialAdjustmentService,
  createWorkerExecutor,
  type SpatialAdjustmentExecutor,
  type SpatialAdjustmentService,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { afterEach, describe, expect, it } from 'vitest';

/** Deterministic mulberry32 so the image is identical on every run. */
const seeded = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

/** Smooth gradients plus noise, with a transparent border, a transparent hole and soft alpha in between. */
const createImage = (width: number, height: number): Uint8ClampedArray => {
  const random = seeded(0x5eed);
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = (y * width + x) * 4;
      const noise = random() * 48 - 24;
      pixels[p] = (x / width) * 200 + noise + 20;
      pixels[p + 1] = (y / height) * 180 + Math.sin(x / 9) * 30 + noise + 30;
      pixels[p + 2] = 120 + Math.cos((x + y) / 13) * 60 + noise;
      const border = x < 12 || y < 10 || x >= width - 20 || y >= height - 8;
      const hole = (x - width / 2) ** 2 + (y - height / 2) ** 2 < 30 ** 2;
      pixels[p + 3] = border || hole ? 0 : x % 37 < 4 ? 96 : 255;
    }
  }
  return pixels;
};

const services: SpatialAdjustmentService[] = [];

const createService = (createExecutor?: () => SpatialAdjustmentExecutor) => {
  const budget = new RasterMemoryBudgetController({ budgetBytes: 512 * 1024 * 1024 });
  const service = createSpatialAdjustmentService({
    createExecutor,
    reserve: (bytes) => budget.reserveOperation(bytes, { purpose: 'layer-operation' }),
  });
  services.push(service);
  return { budget, service };
};

afterEach(() => {
  for (const service of services.splice(0)) {
    service.dispose();
  }
});

describe('spatial adjustment worker', () => {
  it('returns pixels bit-identical to evaluating on the main thread', async () => {
    const { budget, service } = createService();
    const width = 256;
    const height = 192;
    const steps: SpatialAdjustmentStep[] = [
      { amount: 70, kind: 'clarity', size: 24 },
      { amount: 45, kind: 'texture', size: 6 },
      { amount: 35, kind: 'dehaze' },
    ];
    const input = createImage(width, height);
    const expected = input.slice();
    expect(evaluateSpatialSteps(expected, width, height, steps)).toEqual({ status: 'ok' });

    const transferred = input.slice();
    const result = await service.run({ height, pixels: transferred, priority: 'now', slotKey: 'layer', steps, width });

    expect(result.status).toBe('ok');
    if (result.status !== 'ok') {
      return;
    }
    // Zero copy: the caller's buffer moved to the worker and a different one came back.
    expect(transferred.byteLength).toBe(0);
    expect(result.pixels).toEqual(expected);
    expect(result.pixels).not.toEqual(input);
    expect(budget.snapshot().reservedBytes).toBe(0);
  });

  it('reports a job whose buffer cannot be transferred as an error and keeps the same worker', async () => {
    let workers = 0;
    const { service } = createService(() => {
      workers++;
      return createWorkerExecutor();
    });
    const steps: SpatialAdjustmentStep[] = [{ amount: 50, kind: 'clarity', size: 8 }];
    const job = (slotKey: string, pixels: Uint8ClampedArray) =>
      service.run({ height: 32, pixels, priority: 'now', slotKey, steps, width: 32 });

    const first = job('first', createImage(32, 32));
    const detached = createImage(32, 32);
    const broken = job('broken', detached);
    // Contrary to the contract, the caller transfers the queued job's buffer elsewhere.
    structuredClone(detached.buffer, { transfer: [detached.buffer] });
    const after = job('after', createImage(32, 32));

    expect((await first).status).toBe('ok');
    expect(await broken).toEqual({ status: 'error' });
    expect((await after).status).toBe('ok');
    expect(workers).toBe(1);
  });
});

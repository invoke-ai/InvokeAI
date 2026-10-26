import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';

import { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { applyAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { adjustExactly, adjustmentRefusal } from '@workbench/canvas-engine/render/exactAdjustments';
import { evaluateSpatialSteps } from '@workbench/canvas-engine/render/spatial/evaluateSpatialSteps';
import { createSpatialAdjustmentService } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import { createInlineSpatialExecutor } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService.testing';
import { describe, expect, it, vi } from 'vitest';

const WIDTH = 24;
const HEIGHT = 16;
const ORIGIN = { x: -3, y: 5 };

/** A diagonal gradient with a dark block and a transparent border: edges for the kernels, and a sub-domain. */
const source = (): Uint8ClampedArray => {
  const data = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  for (let y = 2; y < HEIGHT - 2; y += 1) {
    for (let x = 2; x < WIDTH - 2; x += 1) {
      const i = (y * WIDTH + x) * 4;
      const dark = x > 8 && x < 14 && y > 5 && y < 10;
      data[i] = dark ? 20 : 60 + x * 6;
      data[i + 1] = dark ? 25 : 90 + y * 5;
      data[i + 2] = dark ? 30 : 200 - x * 4;
      data[i + 3] = 255;
    }
  }
  return data;
};

const BRIGHTEN = { brightness: 0.2, contrast: 0.1, id: 'bc', isEnabled: true, type: 'brightness-contrast' } as const;
const CLARITY = { amount: 80, id: 'cl', isEnabled: true, size: 6, type: 'clarity' } as const;
const GRAIN = {
  amount: 0.2,
  distribution: 'gaussian',
  grain: 'color',
  id: 'grain',
  isEnabled: true,
  response: 'flat',
  seed: 3,
  size: 1,
  type: 'noise',
} as const;
const STACK: CanvasAdjustmentsContract = [BRIGHTEN, CLARITY, GRAIN];

const inlineRunner = () => {
  const memory = new RasterMemoryBudgetController({ budgetBytes: 64 * 1024 * 1024 });
  const service = createSpatialAdjustmentService({
    createExecutor: createInlineSpatialExecutor,
    reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'spatial-adjustment' }),
  });
  return { memory, run: vi.fn(service.run) };
};

describe('adjustExactly', () => {
  it('evaluates the whole stack in list order, per-pixel runs at the origin and spatial runs in the service', async () => {
    const { memory, run } = inlineRunner();

    const result = await adjustExactly({ data: source(), height: HEIGHT, width: WIDTH }, STACK, ORIGIN, { run });

    const expected = source();
    applyAdjustments({ data: expected, height: HEIGHT, width: WIDTH }, [BRIGHTEN], ORIGIN);
    expect(evaluateSpatialSteps(expected, WIDTH, HEIGHT, [{ amount: 80, kind: 'clarity', size: 6 }])).toEqual({
      status: 'ok',
    });
    applyAdjustments({ data: expected, height: HEIGHT, width: WIDTH }, [GRAIN], ORIGIN);
    expect(result.status).toBe('ok');
    expect(result.status === 'ok' && [...result.data]).toEqual([...expected]);
    expect(run).toHaveBeenCalledOnce();
    expect(run.mock.calls[0]![0]).toMatchObject({ priority: 'now' });
    expect(memory.snapshot().reservedBytes).toBe(0);
  });

  it('changes the pixels a per-pixel-only pass leaves, so a spatial entry is never dropped', async () => {
    const { run } = inlineRunner();
    const exact = await adjustExactly({ data: source(), height: HEIGHT, width: WIDTH }, STACK, ORIGIN, { run });
    const perPixel = source();
    applyAdjustments({ data: perPixel, height: HEIGHT, width: WIDTH }, STACK, ORIGIN);

    expect(exact.status === 'ok' && [...exact.data]).not.toEqual([...perPixel]);
  });

  it('applies a stack without spatial entries in place, needing no runner', async () => {
    const data = source();
    const result = await adjustExactly({ data, height: HEIGHT, width: WIDTH }, [BRIGHTEN, GRAIN], ORIGIN, {});

    expect(result).toEqual({ data, status: 'ok' });
  });

  it('refuses a spatial stack without a runner instead of leaving its spatial entries out', async () => {
    await expect(adjustExactly({ data: source(), height: HEIGHT, width: WIDTH }, STACK, ORIGIN, {})).resolves.toEqual({
      status: 'error',
    });
  });

  it.each(['too-large', 'over-budget', 'error', 'aborted'] as const)('passes a %s job through', async (status) => {
    const run = vi.fn(() => Promise.resolve({ status }));

    await expect(
      adjustExactly({ data: source(), height: HEIGHT, width: WIDTH }, STACK, ORIGIN, { run })
    ).resolves.toEqual({ status });
  });

  it('reports a job the service ran out of budget for as over-budget', async () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: WIDTH * HEIGHT * 4 - 1 });
    const service = createSpatialAdjustmentService({
      createExecutor: createInlineSpatialExecutor,
      reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'spatial-adjustment' }),
    });

    await expect(
      adjustExactly({ data: source(), height: HEIGHT, width: WIDTH }, STACK, ORIGIN, { run: service.run })
    ).resolves.toEqual({ status: 'over-budget' });
  });

  it('starts no job once its signal is aborted', async () => {
    const run = vi.fn();
    const controller = new AbortController();
    controller.abort();

    await expect(
      adjustExactly({ data: source(), height: HEIGHT, width: WIDTH }, STACK, ORIGIN, {
        run,
        signal: controller.signal,
      })
    ).resolves.toEqual({ status: 'aborted' });
    expect(run).not.toHaveBeenCalled();
  });
});

describe('adjustmentRefusal', () => {
  const layer = {
    blendMode: 'normal',
    id: 'leaf',
    isEnabled: true,
    isLocked: false,
    name: 'Portrait',
    opacity: 1,
    source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
  } as const;
  const document = {
    background: 'transparent' as const,
    bbox: { height: 8, width: 8, x: 0, y: 0 },
    height: 8,
    selectedLayerId: null,
    stacks: stacksFrom([groupContract('group', [layer], { name: 'Background group' })]),
    version: 3 as const,
    width: 8,
  };

  it('names the layer or group whose stack could not run', () => {
    expect(adjustmentRefusal(document, 'leaf', 'too-large')).toEqual({
      ownerId: 'leaf',
      ownerName: 'Portrait',
      reason: 'too-large',
      status: 'adjustment-refused',
    });
    expect(adjustmentRefusal(document, 'group', 'error')).toMatchObject({ ownerName: 'Background group' });
    expect(adjustmentRefusal(null, 'gone', 'over-budget')).toMatchObject({ ownerName: 'gone' });
  });
});

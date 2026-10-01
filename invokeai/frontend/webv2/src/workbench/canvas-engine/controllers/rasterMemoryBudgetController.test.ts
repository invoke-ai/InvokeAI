import { describe, expect, it } from 'vitest';

import { RasterMemoryBudgetController } from './rasterMemoryBudgetController';

describe('RasterMemoryBudgetController', () => {
  it('reports the currently available bytes used to derive background pixel-area limits', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });
    memory.setCategoryBytes('base', 400);
    memory.setCategoryBytes('decoded', 100);

    expect(memory.getAvailableBytes()).toBe(500);
  });

  it('accounts every allocation class and refuses background reservations beyond the soft limit', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });

    memory.setCategoryBytes('base', 400);
    memory.setCategoryBytes('derived', 150);
    memory.setCategoryBytes('group', 50);
    memory.setCategoryBytes('decoded', 100);
    const detached = memory.trackDetached(50);

    const accepted = memory.reserve(200, { generation: 1, purpose: 'thumbnail' });
    const refused = memory.reserve(100, { generation: 1, purpose: 'raster-export' });

    expect(accepted.status).toBe('ok');
    expect(refused).toEqual({ availableBytes: 50, requestedBytes: 100, status: 'over-budget' });
    expect(memory.snapshot()).toEqual({
      baseBytes: 400,
      decodedBytes: 100,
      derivedBytes: 150,
      detachedBytes: 50,
      groupBytes: 50,
      overageBytes: 0,
      reservedBytes: 200,
      totalBytes: 950,
    });
    detached.release();
  });

  it('reports bytes held above the budget as overage and leaves nothing available', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });
    memory.setCategoryBytes('base', 1_300);

    expect(memory.snapshot().overageBytes).toBe(300);
    expect(memory.reserveOperation(1, { purpose: 'layer-operation' }).status).toBe('over-budget');
  });

  it('makes generation reservation release idempotent across cancellation and disposal', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });
    const reserved = memory.reserve(400, { generation: 7, purpose: 'psd-export' });
    expect(reserved.status).toBe('ok');
    if (reserved.status !== 'ok') {
      throw new Error('Expected reservation');
    }
    const pin = memory.pin('layer-a');

    memory.releaseGeneration(7);
    reserved.lease.release();
    memory.releaseGeneration(7);
    expect(memory.snapshot().reservedBytes).toBe(0);
    expect(memory.isPinned('layer-a')).toBe(true);

    memory.dispose();
    memory.dispose();
    pin.release();
    expect(memory.isPinned('layer-a')).toBe(false);
  });

  it('keeps operation reservations, pins and detached snapshots until their owners release them', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });
    const reservation = memory.reserveOperation(300, { purpose: 'invocation-composite' });
    expect(reservation.status).toBe('ok');
    if (reservation.status !== 'ok') {
      throw new Error('Expected operation reservation');
    }
    const pin = memory.pin('layer-a');
    const detached = memory.trackDetached(200);

    memory.releaseGeneration(4);

    expect(memory.snapshot()).toMatchObject({ detachedBytes: 200, reservedBytes: 300 });
    expect(memory.isPinned('layer-a')).toBe(true);

    reservation.lease.release();
    pin.release();
    detached.release();
    detached.release();
    expect(memory.snapshot()).toMatchObject({ detachedBytes: 0, reservedBytes: 0 });
    expect(memory.isPinned('layer-a')).toBe(false);
  });
});

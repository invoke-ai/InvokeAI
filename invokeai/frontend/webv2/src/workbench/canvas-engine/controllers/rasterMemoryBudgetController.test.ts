import { describe, expect, it } from 'vitest';

import { RasterMemoryBudgetController } from './rasterMemoryBudgetController';

describe('RasterMemoryBudgetController', () => {
  it('reports the currently available bytes used to derive background pixel-area limits', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });
    memory.setBaseBytes(400);
    memory.setDecodedBytes(100);

    expect(memory.getAvailableBytes()).toBe(500);
  });

  it('accounts for owned bytes and refuses background reservations beyond the soft limit', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });

    memory.setBaseBytes(400);
    memory.setDerivedBytes(200);
    memory.setDecodedBytes(100);
    memory.setDetachedBytes(50);

    const accepted = memory.reserve(200, { generation: 1, purpose: 'thumbnail' });
    const refused = memory.reserve(100, { generation: 1, purpose: 'raster-export' });

    expect(accepted.status).toBe('ok');
    expect(refused).toEqual({ availableBytes: 50, requestedBytes: 100, status: 'over-budget' });
    expect(memory.snapshot()).toEqual({
      baseBytes: 400,
      decodedBytes: 100,
      derivedBytes: 200,
      detachedBytes: 50,
      reservedBytes: 200,
      totalBytes: 950,
    });
  });

  it('makes generation reservation and pin release idempotent across cancellation and disposal', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });
    const reserved = memory.reserve(400, { generation: 7, purpose: 'psd-export' });
    expect(reserved.status).toBe('ok');
    if (reserved.status !== 'ok') {
      throw new Error('Expected reservation');
    }
    const pin = memory.pin('layer-a', 7);

    memory.releaseGeneration(7);
    reserved.lease.release();
    pin.release();
    memory.releaseGeneration(7);
    memory.dispose();
    memory.dispose();

    expect(memory.snapshot().reservedBytes).toBe(0);
    expect(memory.isPinned('layer-a')).toBe(false);
  });

  it('keeps caller-owned detached snapshots accounted across generation release', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });
    const detached = memory.trackDetached(300, 9);

    expect(memory.snapshot().detachedBytes).toBe(300);
    memory.releaseGeneration(9);
    expect(memory.snapshot().detachedBytes).toBe(300);
    detached.release();
    expect(memory.snapshot().detachedBytes).toBe(0);
  });

  it('keeps in-flight operation reservations and pins across generation release', () => {
    const memory = new RasterMemoryBudgetController({ budgetBytes: 1_000 });
    const reservation = memory.reserveOperation(300, { purpose: 'invocation-composite' });
    expect(reservation.status).toBe('ok');
    if (reservation.status !== 'ok') {
      throw new Error('Expected operation reservation');
    }
    const pin = memory.pinOperation('layer-a');

    memory.releaseGeneration(4);

    expect(memory.snapshot().reservedBytes).toBe(300);
    expect(memory.isPinned('layer-a')).toBe(true);

    reservation.lease.release();
    pin.release();
    expect(memory.snapshot().reservedBytes).toBe(0);
    expect(memory.isPinned('layer-a')).toBe(false);
  });
});

import { describe, expect, it } from 'vitest';

import { createCanvasEditGate } from './editGate';

describe('createCanvasEditGate', () => {
  it('grants one exclusive lease and makes release idempotently stale', () => {
    const gate = createCanvasEditGate();
    const lease = gate.tryAcquire({ kind: 'filter', layerId: 'a' });
    expect(lease).not.toBeNull();
    expect(gate.tryAcquire({ kind: 'select-object', layerId: 'b' })).toBeNull();
    expect(lease!.isCurrent()).toBe(true);

    lease!.release();
    lease!.release();
    expect(lease!.signal.aborted).toBe(true);
    expect(lease!.isCurrent()).toBe(false);
    expect(gate.tryAcquire({ kind: 'select-object', layerId: 'b' })).not.toBeNull();
  });

  it('invalidates leases on document replacement, project invalidation, cooldown, and disposal', () => {
    const gate = createCanvasEditGate();
    for (const invalidate of [gate.invalidateDocument, gate.invalidateProject]) {
      const lease = gate.tryAcquire({ kind: 'test' });
      expect(lease).not.toBeNull();
      invalidate();
      expect(lease!.signal.aborted).toBe(true);
      expect(lease!.isCurrent()).toBe(false);
    }

    const coolingLease = gate.tryAcquire({ kind: 'cooling' })!;
    gate.cooldown();
    expect(coolingLease.isCurrent()).toBe(false);
    expect(gate.tryAcquire({ kind: 'while-cooling' })).toBeNull();
    gate.activate();

    const disposedLease = gate.tryAcquire({ kind: 'disposed' })!;
    gate.dispose();
    expect(disposedLease.isCurrent()).toBe(false);
    expect(gate.tryAcquire({ kind: 'after-dispose' })).toBeNull();
  });
});

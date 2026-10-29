import type { CanvasProjectMutation } from '@workbench/canvas-engine/api';

import { describe, expect, it, vi } from 'vitest';

import {
  clearCanvasWithRestorePoint,
  restoreFromPoint,
  type RestorePointEngine,
  type RestorePointPorts,
} from './canvasRestorePoints';

interface Fixture {
  open?: boolean;
  blank?: boolean;
  lock?: 'editing' | 'canvas' | null;
  flush?: () => Promise<void>;
  /** What the engine's document swaps report. */
  swaps?: boolean;
}

/** Ports over a recorded project and engine: `order` interleaves flushes, mutations, swaps and stops. */
const createFixture = ({ blank = false, flush, lock = null, open = true, swaps = true }: Fixture = {}) => {
  const order: string[] = [];
  const applied: CanvasProjectMutation[] = [];
  let next = 0;
  const engine: RestorePointEngine = {
    document: {
      replaceDocument: vi.fn(() => {
        order.push('replace');
        return swaps;
      }),
      restoreSnapshot: vi.fn((snapshotId: string, keepCurrentAs?: { id: string; name: string; createdAt: string }) => {
        order.push(`restore ${snapshotId}${keepCurrentAs ? ` keeping ${keepCurrentAs.id}` : ''}`);
        return swaps;
      }),
    },
    lifecycle: {
      flushPendingUploads: vi.fn(async () => {
        order.push('flush');
        await flush?.();
      }),
    },
  };
  const ports: RestorePointPorts = {
    apply: (mutation) => {
      applied.push(mutation);
      order.push(mutation.type);
      return open;
    },
    createId: () => `point-${(next += 1)}`,
    engine: () => (open ? engine : null),
    isBlank: () => blank,
    lockReason: () => lock,
    now: () => '2026-10-29T00:00:00.000Z',
  };
  return { applied, engine, order, ports };
};

const clear = (fixture: ReturnType<typeof createFixture>, stop: string[] = []) => {
  const cancelQueueItem = vi.fn((id: string) => fixture.order.push(`stop ${id}`));
  const outcome = clearCanvasWithRestorePoint({
    cancelQueueItem,
    document: { height: 64, width: 32 },
    name: 'Before clearing the canvas',
    ports: fixture.ports,
    stop,
  });
  return { cancelQueueItem, outcome };
};

describe('clearCanvasWithRestorePoint', () => {
  it('saves the strokes and a restore point with the staged results, stops what it was asked to, then clears', async () => {
    const fixture = createFixture();

    await expect(clear(fixture, ['queue-1']).outcome).resolves.toEqual({
      restorePointId: 'point-1',
      status: 'cleared',
    });
    expect(fixture.engine.lifecycle.flushPendingUploads).toHaveBeenCalledWith({ waitForHeldPixels: false });
    expect(fixture.applied).toEqual([
      {
        createdAt: '2026-10-29T00:00:00.000Z',
        id: 'point-1',
        includeStaging: true,
        kind: 'clear-canvas',
        name: 'Before clearing the canvas',
        type: 'saveCanvasSnapshot',
      },
    ]);
    expect(fixture.order).toEqual(['flush', 'saveCanvasSnapshot', 'stop queue-1', 'replace']);
    expect(vi.mocked(fixture.engine.document.replaceDocument).mock.calls[0]![0]).toMatchObject({
      height: 64,
      width: 32,
    });
  });

  it('clears nothing when the latest strokes cannot be saved', async () => {
    const fixture = createFixture({ flush: () => Promise.reject(new Error('offline')) });
    const { cancelQueueItem, outcome } = clear(fixture, ['queue-1']);

    await expect(outcome).resolves.toEqual({ status: 'unsaved' });
    expect(fixture.applied).toEqual([]);
    expect(cancelQueueItem).not.toHaveBeenCalled();
    expect(fixture.engine.document.replaceDocument).not.toHaveBeenCalled();
  });

  it('keeps no restore point of a blank canvas', async () => {
    const fixture = createFixture({ blank: true });
    await expect(clear(fixture).outcome).resolves.toEqual({ restorePointId: null, status: 'cleared' });
    expect(fixture.order).toEqual(['flush', 'replace']);
  });

  it('withdraws the restore point when the canvas does not take the new document', async () => {
    const fixture = createFixture({ swaps: false });
    await expect(clear(fixture).outcome).resolves.toEqual({ status: 'failed' });
    expect(fixture.applied.at(-1)).toEqual({ snapshotId: 'point-1', type: 'deleteCanvasSnapshot' });
  });

  it('leaves a closed canvas alone', async () => {
    const fixture = createFixture({ open: false });
    await expect(clear(fixture).outcome).resolves.toEqual({ status: 'gone' });
    expect(fixture.order).toEqual([]);
  });
});

describe('restoreFromPoint', () => {
  it('saves the strokes, then keeps the replaced canvas as the point that undoes a canvas restore', async () => {
    const fixture = createFixture();
    await expect(
      restoreFromPoint(fixture.ports, { id: 'p', kind: 'clear-canvas' }, 'Before restoring')
    ).resolves.toEqual({ status: 'restored', undoPointId: 'point-1' });
    expect(fixture.order).toEqual(['flush', 'restore p keeping point-1']);
    expect(fixture.engine.document.restoreSnapshot).toHaveBeenCalledWith('p', {
      createdAt: '2026-10-29T00:00:00.000Z',
      id: 'point-1',
      name: 'Before restoring',
    });
  });

  it('restores nothing when the latest strokes cannot be saved', async () => {
    const fixture = createFixture({ flush: () => Promise.reject(new Error('offline')) });
    await expect(restoreFromPoint(fixture.ports, { id: 'p' }, 'Before')).resolves.toEqual({ status: 'unsaved' });
    expect(fixture.engine.document.restoreSnapshot).not.toHaveBeenCalled();
  });

  it.each(['editing', 'canvas'] as const)('waits while the canvas is locked (%s)', async (reason) => {
    const fixture = createFixture({ lock: reason });
    await expect(restoreFromPoint(fixture.ports, { id: 'p' }, 'Before')).resolves.toEqual({
      reason,
      status: 'locked',
    });
    expect(fixture.order).toEqual([]);
  });

  it('uses the point up without keeping a blank canvas in its place', async () => {
    const fixture = createFixture({ blank: true });
    await expect(restoreFromPoint(fixture.ports, { id: 'p' }, 'Before')).resolves.toEqual({
      status: 'restored',
      undoPointId: null,
    });
    expect(fixture.order).toEqual(['flush', 'restore p', 'deleteCanvasSnapshot']);
  });

  it('re-stages a Discard all point while the canvas is locked, without a flush or keeping the canvas', async () => {
    const fixture = createFixture({ lock: 'canvas' });
    await expect(restoreFromPoint(fixture.ports, { id: 'p', kind: 'discard-results' }, 'Before')).resolves.toEqual({
      status: 'restored',
      undoPointId: null,
    });
    expect(fixture.order).toEqual(['restore p']);
  });

  it('says so when a Discard all point has nothing left to re-stage', async () => {
    const fixture = createFixture({ swaps: false });
    await expect(restoreFromPoint(fixture.ports, { id: 'p', kind: 'discard-results' }, 'Before')).resolves.toEqual({
      status: 'nothing',
    });
  });
});

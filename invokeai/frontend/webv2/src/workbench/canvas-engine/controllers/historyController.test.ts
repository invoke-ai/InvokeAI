import type { HistoryEntry } from '@workbench/canvas-engine/history/history';

import { NO_HELD_ASSET_REFS } from '@workbench/canvas-engine/history/history';
import { describe, expect, it, vi } from 'vitest';

import { HistoryController } from './historyController';

const entry = (label: string, bytes: number, overrides: Partial<HistoryEntry> = {}): HistoryEntry => ({
  bytes,
  heldAssetRefs: NO_HELD_ASSET_REFS,
  label,
  redo: () => undefined,
  undo: () => undefined,
  ...overrides,
});

const record = (controller: HistoryController, recorded: HistoryEntry): void => {
  controller.history.admit(recorded.bytes)!.publish(recorded);
};

describe('HistoryController', () => {
  it('owns history and trims it to the inactive byte budget', () => {
    const controller = new HistoryController({ activeByteBudget: 1_000, inactiveByteBudget: 100 });
    record(controller, entry('old', 60));
    record(controller, entry('new', 70));

    controller.cooldown();

    expect(controller.history.byteSize()).toBe(70);
    expect(controller.history.canUndo()).toBe(true);
    controller.dispose();
    controller.dispose();
    expect(controller.history.byteSize()).toBe(0);
  });

  it('refuses replay while locked or mid-gesture and mirrors undo/redo availability', async () => {
    let canEdit = true;
    let gestureActive = false;
    const canUndo = { set: vi.fn() };
    const canRedo = { set: vi.fn() };
    const controller = new HistoryController({
      canEdit: () => canEdit,
      canRedoStore: canRedo,
      canUndoStore: canUndo,
      isGestureActive: () => gestureActive,
    });
    const undo = vi.fn();
    const redo = vi.fn();
    record(controller, entry('edit', 1, { redo, undo }));

    await expect(controller.undo()).resolves.toEqual({ status: 'applied' });
    expect(undo).toHaveBeenCalledOnce();
    expect(canUndo.set).toHaveBeenLastCalledWith(false);
    expect(canRedo.set).toHaveBeenLastCalledWith(true);

    gestureActive = true;
    await expect(controller.redo()).resolves.toEqual({ status: 'refused' });
    gestureActive = false;
    canEdit = false;
    await expect(controller.redo()).resolves.toEqual({ status: 'refused' });
    expect(redo).not.toHaveBeenCalled();
  });

  it('puts back unrecorded live state before a replay it admits, and not before one it refuses', async () => {
    const order: string[] = [];
    let canEdit = true;
    const controller = new HistoryController({
      beforeReplay: () => order.push('before'),
      canEdit: () => canEdit,
    });
    record(controller, entry('edit', 1, { undo: () => void order.push('undo'), redo: () => void order.push('redo') }));

    await expect(controller.undo()).resolves.toEqual({ status: 'applied' });
    await expect(controller.redo()).resolves.toEqual({ status: 'applied' });
    expect(order).toEqual(['before', 'undo', 'before', 'redo']);

    canEdit = false;
    await expect(controller.undo()).resolves.toEqual({ status: 'refused' });
    expect(order).toHaveLength(4);
  });

  it('reports a failed replay and leaves the step where it was', async () => {
    const reportFailure = vi.fn();
    const controller = new HistoryController({ reportFailure });
    const failure = new Error('pixels unavailable');
    record(
      controller,
      entry('Brush stroke', 1, {
        undo: () => {
          throw failure;
        },
      })
    );

    await expect(controller.undo()).resolves.toMatchObject({ status: 'failed' });

    expect(reportFailure).toHaveBeenCalledWith('Brush stroke', failure);
    expect(controller.history.entries()).toEqual({ future: [], past: ['Brush stroke'] });
  });
});

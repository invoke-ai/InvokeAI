import { describe, expect, it, vi } from 'vitest';

import { HistoryController } from './historyController';

describe('HistoryController', () => {
  it('owns history and trims it to the inactive byte budget', () => {
    const controller = new HistoryController({ activeByteBudget: 1_000, inactiveByteBudget: 100 });
    controller.history.push({ bytes: 60, label: 'old', redo: () => undefined, undo: () => undefined });
    controller.history.push({ bytes: 70, label: 'new', redo: () => undefined, undo: () => undefined });

    controller.cooldown();

    expect(controller.history.byteSize()).toBe(70);
    expect(controller.history.canUndo()).toBe(true);
    controller.dispose();
    controller.dispose();
    expect(controller.history.byteSize()).toBe(0);
  });

  it('owns guarded commands and synchronizes undo/redo stores', () => {
    let canEdit = true;
    let gestureActive = false;
    const canUndo = { set: vi.fn() };
    const canRedo = { set: vi.fn() };
    const endBurst = vi.fn();
    const controller = new HistoryController({
      canEdit: () => canEdit,
      canRedoStore: canRedo,
      canUndoStore: canUndo,
      endBurst,
      isGestureActive: () => gestureActive,
    });
    const undo = vi.fn();
    const redo = vi.fn();
    controller.history.push({ bytes: 1, label: 'edit', redo, undo });

    controller.undo();
    expect(undo).toHaveBeenCalledOnce();
    expect(endBurst).toHaveBeenCalledOnce();
    expect(canUndo.set).toHaveBeenLastCalledWith(false);
    expect(canRedo.set).toHaveBeenLastCalledWith(true);

    gestureActive = true;
    controller.redo();
    expect(redo).not.toHaveBeenCalled();
    gestureActive = false;
    canEdit = false;
    controller.clear();
    expect(controller.history.canRedo()).toBe(true);

    canEdit = true;
    controller.clear();
    expect(controller.history.canRedo()).toBe(false);
    controller.dispose();
  });

  it('isolates store subscriber failures while synchronizing both flags', () => {
    const canUndo = {
      set: vi.fn(() => {
        throw new Error('observer');
      }),
    };
    const canRedo = { set: vi.fn() };
    const controller = new HistoryController({ canRedoStore: canRedo, canUndoStore: canUndo });

    expect(() =>
      controller.history.push({ bytes: 1, label: 'edit', redo: () => undefined, undo: () => undefined })
    ).not.toThrow();
    expect(canRedo.set).toHaveBeenCalled();
    controller.dispose();
  });
});

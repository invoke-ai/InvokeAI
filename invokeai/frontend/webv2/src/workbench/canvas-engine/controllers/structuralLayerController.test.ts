import type { CanvasDocumentContractV2 } from '@workbench/types';

import { createHistory } from '@workbench/canvas-engine/history/history';
import { describe, expect, it, vi } from 'vitest';

import { StructuralLayerController } from './structuralLayerController';

describe('StructuralLayerController', () => {
  it('reports whether a structural commit is currently allowed and committed', () => {
    const dispatch = vi.fn();
    const history = createHistory();
    const controller = new StructuralLayerController({
      canEdit: () => true,
      dispatch,
      getDocument: () => null,
      history,
      isGestureActive: () => false,
    });
    const forward = { id: 'layer', type: 'setCanvasSelectedLayer' } as const;
    const inverse = { id: null, type: 'setCanvasSelectedLayer' } as const;

    expect(controller.canCommit()).toBe(true);
    expect(controller.commit('Select layer', forward, inverse)).toBe(true);
    expect(dispatch).toHaveBeenCalledOnce();
    expect(history.canUndo()).toBe(true);
  });

  it.each([
    { canEdit: false, gestureActive: false, reason: 'editing is disallowed' },
    { canEdit: true, gestureActive: true, reason: 'a gesture is active' },
  ])('reports and refuses a structural commit when $reason', ({ canEdit, gestureActive }) => {
    const dispatch = vi.fn();
    const history = createHistory();
    const controller = new StructuralLayerController({
      canEdit: () => canEdit,
      dispatch,
      getDocument: () => null,
      history,
      isGestureActive: () => gestureActive,
    });

    expect(controller.canCommit()).toBe(false);
    expect(
      controller.commit(
        'Select layer',
        { id: 'layer', type: 'setCanvasSelectedLayer' },
        { id: null, type: 'setCanvasSelectedLayer' }
      )
    ).toBe(false);
    expect(dispatch).not.toHaveBeenCalled();
    expect(history.canUndo()).toBe(false);
  });

  it('reports and refuses structural commits after disposal', () => {
    const dispatch = vi.fn();
    const history = createHistory();
    const controller = new StructuralLayerController({
      canEdit: () => true,
      dispatch,
      getDocument: () => null,
      history,
      isGestureActive: () => false,
    });

    controller.dispose();

    expect(controller.canCommit()).toBe(false);
    expect(
      controller.commit(
        'Select layer',
        { id: 'layer', type: 'setCanvasSelectedLayer' },
        { id: null, type: 'setCanvasSelectedLayer' }
      )
    ).toBe(false);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('owns guarded structural history and coalesces rapid nudges', () => {
    let now = 0;
    const dispatch = vi.fn();
    const history = createHistory();
    const document = {
      layers: [
        {
          id: 'layer',
          isEnabled: true,
          isLocked: false,
          transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        },
      ],
      selectedLayerId: 'layer',
    } as CanvasDocumentContractV2;
    const controller = new StructuralLayerController({
      canEdit: () => true,
      dispatch,
      getDocument: () => document,
      history,
      isGestureActive: () => false,
      now: () => now,
    });

    controller.nudge(1, 0);
    now = 100;
    document.layers[0]!.transform.x = 1;
    controller.nudge(1, 0);

    expect(dispatch).toHaveBeenCalledTimes(2);
    history.undo();
    expect(dispatch).toHaveBeenLastCalledWith(expect.objectContaining({ patch: { transform: { x: 0, y: 0 } } }));
    expect(history.canUndo()).toBe(false);
  });
});

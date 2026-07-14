import type { CanvasDocumentContractV2 } from '@workbench/types';

import { createHistory } from '@workbench/canvas-engine/history/history';
import { describe, expect, it, vi } from 'vitest';

import { StructuralLayerController } from './structuralLayerController';

describe('StructuralLayerController', () => {
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

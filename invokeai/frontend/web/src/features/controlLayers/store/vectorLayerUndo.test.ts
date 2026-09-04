import undoable from 'redux-undo';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  canvasClearHistory,
  canvasSliceConfig,
  canvasUndo,
  vectorLayerAdded,
  vectorLayerPathsReplaced,
  vectorPathAdded,
} from './canvasSlice';

describe('vector layer edit history', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('keeps a canceled edit session out of undo history', () => {
    vi.useFakeTimers();
    vi.stubGlobal('window', { setTimeout });
    const reducer = undoable(canvasSliceConfig.slice.reducer, canvasSliceConfig.undoableConfig?.reduxUndoOptions);
    let history = reducer(undefined, { type: '@@INIT' });
    history = reducer(history, vectorLayerAdded({ isSelected: true }));
    const layer = history.present.vectorLayers.entities[0];
    expect(layer).toBeDefined();
    if (!layer) {
      return;
    }

    const originalPath = {
      id: 'bezier-path-a',
      name: null,
      isClosed: false,
      points: [
        { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null, type: 'corner' as const },
        { anchor: { x: 10, y: 0 }, inHandle: null, outHandle: null, type: 'corner' as const },
      ],
    };
    history = reducer(
      history,
      vectorPathAdded({ entityIdentifier: { id: layer.id, type: 'vector_layer' }, path: originalPath })
    );
    history = reducer(history, canvasClearHistory());

    const undoGroup = 'path-edit-session-a';
    const editedPath = {
      ...originalPath,
      points: [{ ...originalPath.points[0]!, anchor: { x: 5, y: 5 } }, originalPath.points[1]!],
    };
    history = reducer(
      history,
      vectorLayerPathsReplaced({
        entityIdentifier: { id: layer.id, type: 'vector_layer' },
        paths: [editedPath],
        undoGroup,
      })
    );
    vi.advanceTimersByTime(1001);
    history = reducer(
      history,
      vectorLayerPathsReplaced({
        entityIdentifier: { id: layer.id, type: 'vector_layer' },
        paths: [originalPath],
        undoGroup,
      })
    );

    expect(history.present.vectorLayers.entities[0]?.paths).toEqual([originalPath]);
    history = reducer(history, canvasUndo());
    expect(history.present.vectorLayers.entities[0]?.paths).toEqual([originalPath]);
  });

  it('undoes an applied edit session in one step', () => {
    vi.useFakeTimers();
    vi.stubGlobal('window', { setTimeout });
    const reducer = undoable(canvasSliceConfig.slice.reducer, canvasSliceConfig.undoableConfig?.reduxUndoOptions);
    let history = reducer(undefined, { type: '@@INIT' });
    history = reducer(history, vectorLayerAdded({ isSelected: true }));
    const layer = history.present.vectorLayers.entities[0];
    expect(layer).toBeDefined();
    if (!layer) {
      return;
    }

    const originalPath = {
      id: 'bezier-path-a',
      name: null,
      isClosed: false,
      points: [
        { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null, type: 'corner' as const },
        { anchor: { x: 10, y: 0 }, inHandle: null, outHandle: null, type: 'corner' as const },
      ],
    };
    history = reducer(
      history,
      vectorPathAdded({ entityIdentifier: { id: layer.id, type: 'vector_layer' }, path: originalPath })
    );
    history = reducer(history, canvasClearHistory());

    const undoGroup = 'path-edit-session-a';
    for (const x of [2, 4, 6]) {
      history = reducer(
        history,
        vectorLayerPathsReplaced({
          entityIdentifier: { id: layer.id, type: 'vector_layer' },
          paths: [
            {
              ...originalPath,
              points: [{ ...originalPath.points[0]!, anchor: { x, y: x } }, originalPath.points[1]!],
            },
          ],
          undoGroup,
        })
      );
      vi.advanceTimersByTime(1001);
    }

    expect(history.present.vectorLayers.entities[0]?.paths[0]?.points[0]?.anchor).toEqual({ x: 6, y: 6 });
    history = reducer(history, canvasUndo());
    expect(history.present.vectorLayers.entities[0]?.paths).toEqual([originalPath]);
  });
});

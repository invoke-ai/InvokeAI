import undoable from 'redux-undo';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  canvasClearHistory,
  canvasRedo,
  canvasSliceConfig,
  canvasUndo,
  vectorLayerAdded,
  vectorLayerPathsReplaced,
  vectorPathAdded,
  vectorPathTransformed,
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
    vi.advanceTimersByTime(16);
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
      vi.advanceTimersByTime(16);
    }

    expect(history.present.vectorLayers.entities[0]?.paths[0]?.points[0]?.anchor).toEqual({ x: 6, y: 6 });
    history = reducer(history, canvasUndo());
    expect(history.present.vectorLayers.entities[0]?.paths).toEqual([originalPath]);
  });

  it('groups active path transforms with the edit session', () => {
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

    const payload = {
      entityIdentifier: { id: layer.id, type: 'vector_layer' } as const,
      pathId: originalPath.id,
      matrix: [1, 0, 0, 1, 5, 5] as [number, number, number, number, number, number],
      undoGroup: 'path-edit-session-a',
    };
    history = reducer(history, vectorPathTransformed(payload));
    vi.advanceTimersByTime(16);
    history = reducer(history, vectorPathTransformed(payload));

    expect(history.present.vectorLayers.entities[0]?.paths[0]?.points[0]?.anchor).toEqual({ x: 10, y: 10 });
    history = reducer(history, canvasUndo());
    expect(history.present.vectorLayers.entities[0]?.paths).toEqual([originalPath]);
    history = reducer(history, canvasRedo());
    expect(history.present.vectorLayers.entities[0]?.paths[0]?.points[0]?.anchor).toEqual({ x: 10, y: 10 });
  });

  it.each(['redo', 'subsequent action', 'discard'] as const)(
    'preserves the final rapid edit across %s',
    (operation) => {
      vi.useFakeTimers();
      vi.stubGlobal('window', { setTimeout });
      const reducer = undoable(canvasSliceConfig.slice.reducer, canvasSliceConfig.undoableConfig?.reduxUndoOptions);
      let history = reducer(undefined, { type: '@@INIT' });
      history = reducer(history, vectorLayerAdded({ isSelected: true }));
      const entityIdentifier = { type: 'vector_layer' as const, id: history.present.vectorLayers.entities[0]!.id };
      const path = (x: number) => ({
        id: 'path',
        name: null,
        isClosed: false,
        points: [
          { anchor: { x, y: 0 }, inHandle: null, outHandle: null, type: 'corner' as const },
          { anchor: { x: 100, y: 0 }, inHandle: null, outHandle: null, type: 'corner' as const },
        ],
      });
      history = reducer(history, vectorPathAdded({ entityIdentifier, path: path(0) }));
      history = reducer(history, canvasClearHistory());
      for (const x of [1, 2, 3, 4, 5]) {
        history = reducer(history, vectorLayerPathsReplaced({ entityIdentifier, paths: [path(x)], undoGroup: 'edit' }));
        vi.advanceTimersByTime(16);
      }
      if (operation === 'redo') {
        history = reducer(reducer(history, canvasUndo()), canvasRedo());
      } else {
        if (operation === 'discard') {
          history = reducer(
            history,
            vectorLayerPathsReplaced({ entityIdentifier, paths: [path(0)], undoGroup: 'edit' })
          );
        }
        history = reducer(history, vectorPathAdded({ entityIdentifier, path: { ...path(50), id: 'second' } }));
        history = reducer(history, canvasUndo());
      }
      expect(history.present.vectorLayers.entities[0]?.paths[0]?.points[0]?.anchor.x).toBe(
        operation === 'discard' ? 0 : 5
      );
    }
  );
});

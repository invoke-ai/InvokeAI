import { atom } from 'nanostores';
import { describe, expect, it, vi } from 'vitest';

vi.mock('konva', () => {
  class Node {
    add = vi.fn();
    visible = vi.fn();
    destroyChildren = vi.fn();
  }
  return { default: { Group: Node, Path: Node, Rect: Node } };
});
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => prefix,
  offsetCoord: (point: { x: number; y: number }, offset: { x: number; y: number }) => ({
    x: point.x - offset.x,
    y: point.y - offset.y,
  }),
}));

import type { Tool } from 'features/controlLayers/store/types';
import { getBezierPathState, getVectorLayerState } from 'features/controlLayers/store/util';
import type { KonvaEventObject } from 'konva/lib/Node';

import { CanvasPathToolModule } from './CanvasPathToolModule';
import type { CanvasToolModule } from './CanvasToolModule';

const setup = () => {
  const baseTool = atom<Tool>('move');
  const activeTool = atom<Tool>('move');
  const replaceVectorPaths = vi.fn();
  const parent = {
    $baseTool: baseTool,
    $tool: activeTool,
    $cursorPos: atom({ relative: { x: 0, y: 0 } }),
    setBaseTool: vi.fn((tool: Tool) => {
      baseTool.set(tool);
      activeTool.set(tool);
    }),
    clearTemporaryToolHotkeys: vi.fn(),
    manager: {
      buildPath: () => [],
      buildLogger: () => ({ debug: vi.fn() }),
      stateApi: { replaceVectorPaths },
      stage: { unscale: (value: number) => value, getScale: () => 1 },
      getAdapter: vi.fn(() => ({ state: getVectorLayerState('source', { paths: [getBezierPathState('changed')] }) })),
    },
  };
  const module = new CanvasPathToolModule(parent as unknown as CanvasToolModule);
  vi.spyOn(module, 'render').mockImplementation(() => {});
  module.$editSession.set({
    id: 'session',
    entityIdentifier: { id: 'source', type: 'vector_layer' },
    previousBaseTool: 'brush',
    snapshotPaths: [],
    activePathId: 'path',
    activePointIndex: 1,
    selectedPoints: [{ pathId: 'path', pointIndex: 1 }],
    activeHandle: null,
    dragTarget: null,
    history: [],
    historyIndex: 0,
  });
  return { module, parent, replaceVectorPaths };
};

describe('vector edit exit confirmation', () => {
  it('does not dispatch or clear redo when discarding unchanged geometry', () => {
    const { module, parent, replaceVectorPaths } = setup();
    parent.manager.getAdapter.mockReturnValue({ state: getVectorLayerState('source') });
    module.discardEditSession();
    expect(replaceVectorPaths).not.toHaveBeenCalled();
    expect(module.$editSession.get()).toBeNull();
  });

  it('clears the drag target when the pointer is interrupted', () => {
    const { module, replaceVectorPaths } = setup();
    module.$editSession.set({
      ...module.$editSession.get()!,
      dragTarget: { type: 'anchor', pathId: 'path', pointIndex: 1 },
    });
    module.onWindowPointerUp();
    expect(module.$editSession.get()?.dragTarget).toBeNull();
    module.onWindowPointerMove({ buttons: 0 } as PointerEvent);
    expect(replaceVectorPaths).not.toHaveBeenCalled();
  });

  it.each([-9, 109, 50])('only inserts inside a segment on Shift-click at x=%s', (x) => {
    const { module, parent, replaceVectorPaths } = setup();
    const path = getBezierPathState('path', {
      points: [
        { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: 100, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
      ],
    });
    parent.manager.getAdapter.mockReturnValue({ state: getVectorLayerState('source', { paths: [path] }) });
    vi.spyOn(module, 'getCanMutateEditSession').mockReturnValue(true);
    parent.$cursorPos.set({ relative: { x, y: 0 } });
    module.onStagePointerDown({ evt: { button: 0, shiftKey: true } } as KonvaEventObject<PointerEvent>);
    if (x === 50) {
      expect(replaceVectorPaths).toHaveBeenCalledOnce();
      expect(replaceVectorPaths.mock.calls[0]![0].paths[0].points).toHaveLength(3);
    } else {
      expect(replaceVectorPaths).not.toHaveBeenCalled();
    }
  });
  it('cancels a pending switch without discarding edits or selection', () => {
    const { module, replaceVectorPaths } = setup();
    const session = module.$editSession.get();
    const switchLayer = vi.fn();
    module.requestEditExit(switchLayer);
    expect(module.$isExitConfirmationOpen.get()).toBe(true);
    expect(switchLayer).not.toHaveBeenCalled();
    module.cancelToolChange();
    module.confirmEditExit('apply');
    expect(module.$editSession.get()).toBe(session);
    expect(module.$isExitConfirmationOpen.get()).toBe(false);
    expect(switchLayer).not.toHaveBeenCalled();
    expect(replaceVectorPaths).not.toHaveBeenCalled();
  });

  it.each(['apply', 'discard'] as const)('resolves %s before performing the pending switch once', (decision) => {
    const { module, replaceVectorPaths } = setup();
    const switchLayer = vi.fn(() => {
      expect(module.$editSession.get()).toBeNull();
      expect(module.$isExitConfirmationOpen.get()).toBe(false);
      expect(replaceVectorPaths).toHaveBeenCalledTimes(decision === 'discard' ? 1 : 0);
    });
    module.requestEditExit(switchLayer);
    module.confirmEditExit(decision);
    module.confirmEditExit(decision);
    expect(switchLayer).toHaveBeenCalledTimes(1);
    if (decision === 'discard') {
      expect(replaceVectorPaths).toHaveBeenCalledWith({
        entityIdentifier: { id: 'source', type: 'vector_layer' },
        paths: [],
        undoGroup: 'session',
      });
    }
  });

  it('keeps the first requested destination while the dialog is open', () => {
    const { module } = setup();
    const first = vi.fn();
    const second = vi.fn();
    module.requestEditExit(first);
    module.requestEditExit(second);
    module.confirmEditExit('apply');
    expect(first).toHaveBeenCalledOnce();
    expect(second).not.toHaveBeenCalled();
  });

  it('defers an eraser switch and restores it only after confirmation', () => {
    const { module, parent } = setup();
    parent.setBaseTool('eraser');
    module.onToolChanged();
    expect(module.$isExitConfirmationOpen.get()).toBe(true);
    expect(parent.$baseTool.get()).toBe('move');
    module.confirmEditExit('apply');
    expect(parent.$baseTool.get()).toBe('eraser');
    expect(module.$editSession.get()).toBeNull();
  });

  it('cancels a requested tool switch and keeps the editing tool', () => {
    const { module, parent } = setup();
    const session = module.$editSession.get();
    parent.setBaseTool('brush');
    module.onToolChanged();
    module.cancelToolChange();
    expect(parent.$baseTool.get()).toBe('move');
    expect(module.$editSession.get()).toBe(session);
    expect(module.$isExitConfirmationOpen.get()).toBe(false);
  });

  it.each(['move', 'rect', 'view', 'colorPicker'] as const)('preserves editing with %s without prompting', (tool) => {
    const { module, parent } = setup();
    const session = module.$editSession.get();
    if (tool === 'move' || tool === 'rect') {
      parent.setBaseTool(tool);
    } else {
      parent.$tool.set(tool);
    }
    module.onToolChanged();
    expect(module.$editSession.get()).toBe(session);
    expect(module.$isExitConfirmationOpen.get()).toBe(false);
  });

  it('executes requests immediately outside Edit mode', () => {
    const { module } = setup();
    module.$editSession.set(null);
    const action = vi.fn();
    module.requestEditExit(action);
    expect(action).toHaveBeenCalledOnce();
    expect(module.$isExitConfirmationOpen.get()).toBe(false);
  });
});

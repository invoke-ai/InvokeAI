import type { CanvasEngine, CanvasInteractionState } from '@workbench/canvas-engine/api';

import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createShortcutHintSources } from '@workbench/hotkeys/hintSources';
import { describe, expect, it, vi } from 'vitest';

import { createCanvasShortcutHintSource, getCanvasShortcutHints } from './shortcutHints';

vi.mock('@workbench/canvas-operations/api', () => ({
  getCanvasOperations: () => ({
    getOperationState: () => ({ status: 'idle' }),
    getSamSessionState: () => null,
    subscribeOperation: () => () => {},
    subscribeSamSession: () => () => {},
  }),
}));

const state = {
  activeTool: 'brush' as const,
  canUndo: false,
  hasSelection: false,
  hasTransform: false,
  isLocked: false,
  lassoShape: 'freehand',
  operation: null,
  samVisual: false,
  shapeKind: 'rect',
};
const hintLabels = (input: Parameters<typeof getCanvasShortcutHints>[0]) =>
  getCanvasShortcutHints(input).hints.map((hint) => ('commandId' in hint ? hint.commandId : hint.labelKey));

describe('Canvas gesture guidance', () => {
  it('teaches the active View gestures without the temporary-tool Pan duplicate', () => {
    expect(getCanvasShortcutHints({ ...state, activeTool: 'view' }).hints).toEqual([
      { labelKey: 'workbench.shortcuts.actions.pan', parts: [], pointerKey: 'workbench.shortcuts.pointer.drag' },
      { labelKey: 'workbench.shortcuts.actions.zoom', parts: [], pointerKey: 'workbench.shortcuts.pointer.scroll' },
    ]);
  });

  it('keeps one registered source through lock changes, caches equal inputs and releases all notifications', () => {
    const stores = createEngineStores('brush');
    const engine = {
      interaction: {
        get: (key: keyof CanvasInteractionState) => stores[key].get(),
        subscribe: (key: keyof CanvasInteractionState, listener: () => void) => stores[key].subscribe(listener),
      },
    } as unknown as CanvasEngine;
    const locked = createExternalStoreCore({ value: false });
    const source = createCanvasShortcutHintSource(engine, 'one', 'canvas', {
      getSnapshot: () => locked.getSnapshot().value,
      subscribe: locked.subscribe,
    });
    const sources = createShortcutHintSources();
    const release = sources.register(source);
    let notifications = 0;
    const unsubscribe = sources.subscribe(() => notifications++);
    const brush = source.getSnapshot();
    expect(source.getSnapshot()).toBe(brush);
    stores.lassoOptions.set({ shape: 'freehand', mode: 'add' });
    expect(source.getSnapshot()).toBe(brush);
    locked.setSnapshot({ value: true });
    expect(sources.get('one', 'canvas')).toBe(source);
    expect(source.getSnapshot().titleKey).toBe('widgets.canvas.tools.view');
    expect(source.getSnapshot().hints).toHaveLength(2);
    locked.setSnapshot({ value: false });
    expect(source.getSnapshot().titleKey).toBe('widgets.canvas.tools.brush');
    release();
    expect(sources.get('one', 'canvas')).toBeNull();
    const afterRelease = notifications;
    locked.setSnapshot({ value: true });
    stores.activeTool.set('view');
    expect(notifications).toBe(afterRelease);
    unsubscribe();
  });
  it('removes painting/session instructions during staging and prompt-mode object selection', () => {
    expect(hintLabels({ ...state, isLocked: true })).toEqual([
      'workbench.shortcuts.actions.pan',
      'workbench.shortcuts.actions.zoom',
    ]);
    expect(hintLabels({ ...state, operation: 'select-object', samVisual: false })).toEqual([
      'workbench.shortcuts.actions.pan',
      'workbench.shortcuts.actions.zoom',
    ]);
    expect(hintLabels({ ...state, operation: 'filter', hasSelection: true, canUndo: true })).not.toContain(
      'canvas.undo'
    );
  });

  it('prioritizes active transform completion and teaches marquee modifier timing without sampling', () => {
    expect(hintLabels({ ...state, activeTool: 'transform', hasTransform: true }).slice(0, 2)).toEqual([
      'workbench.shortcuts.actions.apply',
      'workbench.shortcuts.actions.cancel',
    ]);
    expect(hintLabels({ ...state, activeTool: 'transform', hasTransform: false })).not.toContain(
      'workbench.shortcuts.actions.apply'
    );
    const marquee = hintLabels({ ...state, activeTool: 'marquee' });
    expect(marquee).toContain('workbench.shortcuts.actions.addBeforeDrag');
    expect(marquee).toContain('workbench.shortcuts.actions.constrainDuringDrag');
    expect(marquee).not.toContain('workbench.shortcuts.actions.sample');
  });

  it('replaces freehand instructions for polygons and restores selection/history commands only when available', () => {
    const polygon = hintLabels({
      ...state,
      activeTool: 'lasso',
      lassoShape: 'polygon',
      canUndo: true,
      hasSelection: true,
    });
    expect(polygon.slice(0, 3)).toEqual([
      'workbench.shortcuts.actions.placePoint',
      'workbench.shortcuts.actions.closePolygon',
      'workbench.shortcuts.actions.cancel',
    ]);
    expect(polygon).toContain('canvas.deselect');
    expect(polygon).toContain('canvas.undo');
    const sam = hintLabels({ ...state, operation: 'select-object', samVisual: true, canUndo: true });
    expect(sam).toContain('workbench.shortcuts.actions.samOppositePoint');
    expect(sam).not.toContain('canvas.undo');
    expect(sam).not.toContain('workbench.shortcuts.actions.apply');
  });
});

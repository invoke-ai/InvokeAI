import type { CanvasEngine, CanvasInteractionState, ToolId } from '@workbench/canvas-engine/api';
import type { ShortcutHint, ShortcutHintSnapshot, ShortcutHintSource } from '@workbench/hotkeys/hintSources';

import { getCanvasOperations } from '@workbench/canvas-operations/api';

const gesture = (label: string, parts: readonly string[] = [], pointerKey?: string): ShortcutHint => ({
  labelKey: `workbench.shortcuts.actions.${label}`,
  parts,
  pointerKey: pointerKey ? `workbench.shortcuts.pointer.${pointerKey}` : undefined,
});
const PAN = gesture('pan', ['space'], 'drag');
const ZOOM = gesture('zoom', [], 'scroll');
const SAMPLE = gesture('sample', ['alt'], 'click');
const CANCEL = gesture('cancel', ['esc']);
const POLYGON_HINTS = [gesture('placePoint', [], 'click'), gesture('closePolygon', ['enter']), CANCEL];

/** Fixed gestures follow engine tools; configurable commands are resolved by the Workbench host. */
export const getCanvasShortcutHints = ({
  activeTool,
  lassoShape,
  shapeKind,
  hasTransform,
  hasSelection,
  canUndo,
  isLocked,
  operation,
  samVisual,
}: {
  activeTool: ToolId;
  lassoShape: string;
  shapeKind: string;
  hasTransform: boolean;
  hasSelection: boolean;
  canUndo: boolean;
  isLocked: boolean;
  operation: 'filter' | 'select-object' | null;
  samVisual: boolean;
}): ShortcutHintSnapshot => {
  const hints: ShortcutHint[] = [];
  if (isLocked || operation === 'filter' || (operation === 'select-object' && !samVisual)) {
    return { hints: [PAN, ZOOM], titleKey: 'widgets.canvas.tools.view' };
  }
  const tool = operation === 'select-object' ? 'sam' : activeTool;
  if (tool === 'sam') {
    hints.push(
      gesture('samPoint', [], 'click'),
      gesture('samOppositePoint', ['shift'], 'click'),
      gesture('samRemovePoint', [], 'point'),
      gesture('samBox', [], 'drag')
    );
  } else if (tool === 'transform' && hasTransform) {
    hints.push(
      gesture('apply', ['enter']),
      CANCEL,
      gesture('constrainTransform', ['shift'], 'drag'),
      gesture('scaleCenter', ['alt'], 'handleDrag')
    );
  } else if (tool === 'marquee') {
    hints.push(
      gesture('select', [], 'drag'),
      gesture('addBeforeDrag', ['shift'], 'drag'),
      gesture('subtractBeforeDrag', ['alt'], 'drag'),
      gesture('constrainDuringDrag', ['shift']),
      gesture('centerDuringDrag', ['alt']),
      CANCEL
    );
  } else if (tool === 'lasso') {
    hints.push(
      ...(lassoShape === 'polygon' ? POLYGON_HINTS : [gesture('select', [], 'drag'), CANCEL]),
      gesture('addSelection', ['shift']),
      gesture('subtractSelection', ['alt'])
    );
  } else if (tool === 'shape') {
    hints.push(
      ...(shapeKind === 'polygon'
        ? POLYGON_HINTS
        : [
            gesture('draw', [], 'drag'),
            ...(shapeKind === 'freehand' ? [] : [gesture('constrain', ['shift'], 'drag')]),
            CANCEL,
          ])
    );
  } else if (tool === 'view') {
    hints.push(gesture('pan', [], 'drag'), ZOOM);
  } else if (tool === 'colorPicker') {
    hints.push(gesture('sample', [], 'click'));
  } else if (tool === 'brush' || tool === 'eraser') {
    hints.push(
      { commandId: 'canvas.brushSizeDown', labelKey: 'workbench.shortcuts.actions.smaller' },
      { commandId: 'canvas.brushSizeUp', labelKey: 'workbench.shortcuts.actions.larger' },
      SAMPLE
    );
  }
  if (tool !== 'view') {
    hints.push(PAN, ZOOM);
  }
  if (hasSelection && !operation) {
    hints.push({ commandId: 'canvas.deselect', labelKey: 'workbench.shortcuts.actions.deselect' });
  }
  if (canUndo && !operation) {
    hints.push({ commandId: 'canvas.undo', labelKey: 'workbench.shortcuts.actions.undo' });
  }
  return { hints, titleKey: `widgets.canvas.tools.${tool}` };
};

const HINT_STATE_KEYS: readonly (keyof CanvasInteractionState)[] = [
  'activeTool',
  'lassoOptions',
  'shapeOptions',
  'transformSession',
  'hasFloatingSelection',
  'hasSelection',
  'canUndo',
];

/** Observe an existing engine only. No lease, pixel/document epoch, pointer coordinate or parameter subscriptions. */
export const createCanvasShortcutHintSource = (
  engine: CanvasEngine,
  projectId: string,
  instanceId: string,
  lock: { getSnapshot(): boolean; subscribe(listener: () => void): () => void }
): ShortcutHintSource => {
  const operations = getCanvasOperations(engine);
  let snapshot: ShortcutHintSnapshot | null = null;
  let inputs: Parameters<typeof getCanvasShortcutHints>[0] | null = null;
  return {
    projectId,
    instanceId,
    getSnapshot: () => {
      const operation = operations.getOperationState();
      const sam = operations.getSamSessionState();
      const nextInputs = {
        activeTool: engine.interaction.get('activeTool'),
        canUndo: engine.interaction.get('canUndo'),
        hasSelection: engine.interaction.get('hasSelection'),
        hasTransform:
          engine.interaction.get('transformSession') !== null || engine.interaction.get('hasFloatingSelection'),
        isLocked: lock.getSnapshot(),
        lassoShape: engine.interaction.get('lassoOptions').shape,
        operation: operation.status === 'active' ? operation.identity.kind : null,
        samVisual: sam?.input.type === 'visual' && sam.status !== 'committing',
        shapeKind: engine.interaction.get('shapeOptions').kind,
      } satisfies Parameters<typeof getCanvasShortcutHints>[0];
      if (
        !snapshot ||
        !inputs ||
        (Object.keys(nextInputs) as (keyof typeof nextInputs)[]).some((key) => inputs![key] !== nextInputs[key])
      ) {
        snapshot = getCanvasShortcutHints(nextInputs);
        inputs = nextInputs;
      }
      return snapshot;
    },
    subscribe: (listener) => {
      const unsubscribes = [
        ...HINT_STATE_KEYS.map((key) => engine.interaction.subscribe(key, listener)),
        operations.subscribeOperation(listener),
        operations.subscribeSamSession(listener),
        lock.subscribe(listener),
      ];
      return () => unsubscribes.forEach((unsubscribe) => unsubscribe());
    },
  };
};

import type { Tool } from 'features/controlLayers/store/types';

type CanvasToolModifierHintKey = 'mod' | 'shift' | 'alt' | 'space' | 'wheel' | 'arrows' | 'enter' | 'esc';

type CanvasToolModifierHintId =
  | 'spacePan'
  | 'altPickColor'
  | 'shiftStraightLine'
  | 'modWheelResizeBrush'
  | 'modWheelResizeEraser'
  | 'modSubtractMask'
  | 'shiftSnap45Degrees'
  | 'shiftLockAspectRatio'
  | 'shiftUnlockAspectRatio'
  | 'altScaleFromCenter'
  | 'modFineGrid'
  | 'enterCommitText'
  | 'shiftEnterNewLine'
  | 'escCancelText'
  | 'modDragText'
  | 'shiftSnapRotation'
  | 'arrowKeysNudgeSelection';

type CanvasToolModifierHint = {
  id: CanvasToolModifierHintId;
  keys: CanvasToolModifierHintKey[];
  labelKey: string;
};

const SHARED_HINT_IDS = ['spacePan', 'altPickColor'] as const satisfies readonly CanvasToolModifierHintId[];

const HINTS: Record<CanvasToolModifierHintId, CanvasToolModifierHint> = {
  spacePan: {
    id: 'spacePan',
    keys: ['space'],
    labelKey: 'controlLayers.modifierHints.labels.pan',
  },
  altPickColor: {
    id: 'altPickColor',
    keys: ['alt'],
    labelKey: 'controlLayers.modifierHints.labels.pickColor',
  },
  shiftStraightLine: {
    id: 'shiftStraightLine',
    keys: ['shift'],
    labelKey: 'controlLayers.modifierHints.labels.straightLine',
  },
  modWheelResizeBrush: {
    id: 'modWheelResizeBrush',
    keys: ['mod', 'wheel'],
    labelKey: 'controlLayers.modifierHints.labels.resizeBrush',
  },
  modWheelResizeEraser: {
    id: 'modWheelResizeEraser',
    keys: ['mod', 'wheel'],
    labelKey: 'controlLayers.modifierHints.labels.resizeEraser',
  },
  modSubtractMask: {
    id: 'modSubtractMask',
    keys: ['mod'],
    labelKey: 'controlLayers.modifierHints.labels.subtractMask',
  },
  shiftSnap45Degrees: {
    id: 'shiftSnap45Degrees',
    keys: ['shift'],
    labelKey: 'controlLayers.modifierHints.labels.snap45Degrees',
  },
  shiftLockAspectRatio: {
    id: 'shiftLockAspectRatio',
    keys: ['shift'],
    labelKey: 'controlLayers.modifierHints.labels.lockAspectRatio',
  },
  shiftUnlockAspectRatio: {
    id: 'shiftUnlockAspectRatio',
    keys: ['shift'],
    labelKey: 'controlLayers.modifierHints.labels.unlockAspectRatio',
  },
  altScaleFromCenter: {
    id: 'altScaleFromCenter',
    keys: ['alt'],
    labelKey: 'controlLayers.modifierHints.labels.scaleFromCenter',
  },
  modFineGrid: {
    id: 'modFineGrid',
    keys: ['mod'],
    labelKey: 'controlLayers.modifierHints.labels.fineGrid',
  },
  enterCommitText: {
    id: 'enterCommitText',
    keys: ['enter'],
    labelKey: 'controlLayers.modifierHints.labels.commitText',
  },
  shiftEnterNewLine: {
    id: 'shiftEnterNewLine',
    keys: ['shift', 'enter'],
    labelKey: 'controlLayers.modifierHints.labels.newLine',
  },
  escCancelText: {
    id: 'escCancelText',
    keys: ['esc'],
    labelKey: 'controlLayers.modifierHints.labels.cancelText',
  },
  modDragText: {
    id: 'modDragText',
    keys: ['mod'],
    labelKey: 'controlLayers.modifierHints.labels.dragText',
  },
  shiftSnapRotation: {
    id: 'shiftSnapRotation',
    keys: ['shift'],
    labelKey: 'controlLayers.modifierHints.labels.snapRotation',
  },
  arrowKeysNudgeSelection: {
    id: 'arrowKeysNudgeSelection',
    keys: ['arrows'],
    labelKey: 'controlLayers.modifierHints.labels.nudgeSelection',
  },
};

type GetCanvasToolModifierHintsArg = {
  tool: Tool;
  lassoMode: 'freehand' | 'polygon';
  bboxAspectRatioLocked: boolean;
  hasActiveTextSession: boolean;
};

const mapHintIdsToHints = (hintIds: readonly CanvasToolModifierHintId[]): CanvasToolModifierHint[] =>
  hintIds.map((hintId) => HINTS[hintId]);

export const getCanvasToolModifierHintIds = ({
  tool,
  lassoMode,
  bboxAspectRatioLocked,
  hasActiveTextSession,
}: GetCanvasToolModifierHintsArg): CanvasToolModifierHintId[] => {
  // Resolver map: each tool returns the relevant hint ids based on the provided args.
  const TOOL_HINT_RESOLVERS: Record<
    Tool,
    (args: GetCanvasToolModifierHintsArg) => readonly CanvasToolModifierHintId[]
  > = {
    brush: () => ['shiftStraightLine', 'modWheelResizeBrush', ...SHARED_HINT_IDS],
    eraser: () => ['shiftStraightLine', 'modWheelResizeEraser', 'spacePan'],
    lasso: ({ lassoMode: lm }) =>
      lm === 'polygon' ? ['modSubtractMask', 'shiftSnap45Degrees', 'spacePan'] : ['modSubtractMask', 'spacePan'],
    bbox: ({ bboxAspectRatioLocked: locked }) => [
      locked ? 'shiftUnlockAspectRatio' : 'shiftLockAspectRatio',
      'altScaleFromCenter',
      'modFineGrid',
    ],
    move: () => ['arrowKeysNudgeSelection', ...SHARED_HINT_IDS],
    text: ({ hasActiveTextSession: active }) =>
      active
        ? ['enterCommitText', 'shiftEnterNewLine', 'escCancelText', 'modDragText', 'shiftSnapRotation']
        : [...SHARED_HINT_IDS],
    view: () => ['altPickColor'],
    colorPicker: () => ['spacePan'],
    gradient: () => [...SHARED_HINT_IDS],
    rect: () => [...SHARED_HINT_IDS],
  };

  const resolver = TOOL_HINT_RESOLVERS[tool];
  // Guard at runtime: if a resolver is missing, return an empty array instead of throwing.
  if (!resolver) {
    return [];
  }
  return Array.from(resolver({ tool, lassoMode, bboxAspectRatioLocked, hasActiveTextSession }));
};

export const getCanvasToolModifierHints = (args: GetCanvasToolModifierHintsArg): CanvasToolModifierHint[] =>
  mapHintIdsToHints(getCanvasToolModifierHintIds(args));

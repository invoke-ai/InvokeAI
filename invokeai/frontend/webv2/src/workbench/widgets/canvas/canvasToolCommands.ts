import type {
  AdjustmentRefusal,
  BeginTransformResult,
  CanvasDocumentContractV3,
  CanvasEngine,
  DocumentRefusal,
  LayerUnreadable,
  ShapeToolKind,
} from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';
import type { ElementType } from 'react';

import { captureAccountScope } from '@platform/state/accountLifecycle';
import { createActionToast } from '@platform/ui/toaster';
import { getDocumentNode, mergeDownEligibility } from '@workbench/canvas-engine/api';
import { reportLayerOperation, type LayerOperationRefusal } from '@workbench/widgets/canvas/useStructuralCommit';
import { CircleIcon, PenLineIcon, PentagonIcon, SquareIcon, StarIcon, TriangleIcon } from 'lucide-react';

/**
 * A command refusal the user can fix, shown the way the Canvas shows its edit refusals: an info toast only, never
 * recorded in Notifications, and one per `key`, so pressing the key again (or holding it) replaces the toast rather
 * than stacking another. It lives here, beside the commands that raise it, because the Canvas and its companion
 * widgets already load this module; a module of its own would cost Canvas activation a request.
 */
export const showCanvasRefusalNotice = (key: string, title: string, message: string): void => {
  createActionToast({
    actions: [],
    description: message,
    id: `canvas-refusal:${key}`,
    signal: captureAccountScope().signal,
    title,
    type: 'info',
  });
};

/** Shows a refusal notice under `key`; see {@link showCanvasRefusalNotice}. */
export type CanvasRefusalNotice = typeof showCanvasRefusalNotice;

/** One hotkey row: its command id, title key and default keys. */
export type HotkeyTableEntry = readonly [id: string, titleKey: string, defaultKeys: readonly string[]];

/** The same layer operation is available from the Canvas and the Layers widget. */
export const CANVAS_MERGE_DOWN_HOTKEY = ['canvas.mergeDown', 'widgets.canvas.commands.mergeDown', ['mod+e']] as const;

/**
 * Merges the selected layer into the one below it. A layer that cannot merge is explained through `reportIneligible`
 * (nothing selected stays silent); a merge the engine refuses or that fails goes to `report`.
 */
export const mergeSelectedLayerDown = (
  engine: Pick<CanvasEngine, 'layers'> | null,
  document: CanvasDocumentContractV3,
  report: (refusal: LayerOperationRefusal | AdjustmentRefusal | LayerUnreadable) => void,
  reportIneligible: (refusal: DocumentRefusal) => void
): void => {
  const id = document.selectedLayerId;
  if (!engine || !id) {
    return;
  }
  const eligibility = mergeDownEligibility(document, id);
  if (eligibility.status !== 'eligible') {
    reportIneligible(eligibility);
    return;
  }
  void engine.layers.mergeLayerDown(id).then(
    (result) => {
      if (result !== 'merged') {
        report(result);
      }
    },
    () => report('failed')
  );
};

/** Merge down's own invalid-target reasons (`mergeDownEligibility`); the name is the layer or group they concern. */
const MERGE_DOWN_TARGET_KEYS: Partial<
  Record<Extract<DocumentRefusal, { status: 'invalid-target' }>['reason'], string>
> = {
  'bottom-of-group': 'widgets.layers.actions.mergeDownBottomOfGroup',
  disabled: 'widgets.layers.actions.mergeDownHidden',
  'group-below': 'widgets.layers.actions.mergeDownGroupBelow',
  'no-layer-below': 'widgets.layers.actions.mergeDownNoLayerBelow',
  'not-mergeable': 'widgets.layers.actions.mergeDownNotMergeable',
};

/**
 * Says why the selected layer cannot merge down, naming the layer or group in the way, as one notice per layer and
 * reason. The engine decides the reason; this only words it.
 */
export const reportMergeDownRefusal = (
  document: CanvasDocumentContractV3,
  refusal: DocumentRefusal,
  notice: CanvasRefusalNotice,
  t: TFunction
): void => {
  const selectedId = document.selectedLayerId;
  if (!selectedId) {
    return;
  }
  const nameOf = (id: string | undefined): string => getDocumentNode(document, id)?.name ?? '';
  const message = ((): string | null => {
    switch (refusal.status) {
      case 'locked':
        return t('widgets.layers.actions.mergeDownLocked', { name: nameOf(refusal.ids[0]) });
      case 'wrong-type':
        return t(
          refusal.actual === 'group'
            ? 'widgets.layers.actions.mergeDownGroupSelected'
            : refusal.actual === 'inpaint_mask'
              ? 'widgets.layers.actions.mergeDownMaskSelected'
              : 'widgets.layers.actions.mergeDownWrongType'
        );
      case 'invalid-target': {
        const key = MERGE_DOWN_TARGET_KEYS[refusal.reason];
        return key ? t(key, { name: nameOf(refusal.targetId) }) : null;
      }
      case 'missing':
      case 'unsupported':
        // The selection is gone, or the command does not exist here: nothing the user can act on.
        return null;
    }
  })();
  if (message !== null) {
    const reason = refusal.status === 'invalid-target' ? refusal.reason : refusal.status;
    notice(
      `merge-down:${selectedId}:${reason}`,
      t('widgets.layers.actions.mergeDownRefused', { name: nameOf(selectedId) }),
      message
    );
  }
};

/**
 * Canvas keys that also act while a companion widget (Layers, with its Color pane) holds focus: tools, the
 * transform session, brush size and the color pair. The Canvas and the companion both register from this table;
 * companions import this module alone, without the Canvas's full dispatcher (`canvasHotkeyCommands.ts`).
 */
export const CANVAS_COMPANION_HOTKEYS = [
  ['canvas.tool.view', 'widgets.canvas.commands.selectViewTool', ['h']],
  ['canvas.tool.move', 'widgets.canvas.commands.selectMoveTool', ['v']],
  ['canvas.transformSelected', 'widgets.canvas.commands.selectTransformTool', ['shift+t']],
  ['canvas.tool.bbox', 'widgets.canvas.commands.selectBboxTool', []],
  ['canvas.tool.brush', 'widgets.canvas.commands.selectBrushTool', ['b']],
  ['canvas.tool.eraser', 'widgets.canvas.commands.selectEraserTool', ['e']],
  ['canvas.tool.lasso', 'widgets.canvas.commands.selectLassoTool', ['l']],
  // M and U as in other image editors: Marquee, then the Shape tool family.
  ['canvas.tool.marquee', 'widgets.canvas.commands.selectMarqueeTool', ['m']],
  ['canvas.tool.shape', 'widgets.canvas.commands.selectShapeTool', ['u']],
  ['canvas.tool.text', 'widgets.canvas.commands.selectTextTool', ['t']],
  ['canvas.tool.gradient', 'widgets.canvas.commands.selectGradientTool', ['g']],
  // A sticky picker for those who would rather not hold Alt; Alt still picks from tools that do not read it.
  ['canvas.tool.colorPicker', 'widgets.canvas.commands.selectColorPickerTool', ['i']],
  ['canvas.brushSizeDown', 'widgets.canvas.commands.decreaseBrushSize', ['[']],
  ['canvas.brushSizeUp', 'widgets.canvas.commands.increaseBrushSize', [']']],
  // The active color pair: X swaps, D resets to black/white.
  ['canvas.toggleFillColor', 'widgets.canvas.commands.swapColors', ['x']],
  ['canvas.setFillColorsToDefault', 'widgets.canvas.commands.resetColors', ['d']],
] as const satisfies readonly HotkeyTableEntry[];

/** Canvas history, which the Canvas and its companion widgets both bind. */
export const CANVAS_HISTORY_HOTKEYS = [
  ['canvas.undo', 'widgets.canvas.commands.undo', ['mod+z']],
  ['canvas.redo', 'widgets.canvas.commands.redo', ['mod+shift+z', 'mod+y']],
] as const satisfies readonly HotkeyTableEntry[];

/**
 * How each shape kind is shown wherever the next shape's kind is chosen: the tool strip flyout and the Shape
 * options. It lives with the Shape key's command so the Canvas and the Layers pane share it from a chunk both load.
 */
export const SHAPE_KIND_PRESENTATION: Record<ShapeToolKind, { icon: ElementType; labelKey: string }> = {
  ellipse: { icon: CircleIcon, labelKey: 'widgets.canvas.toolOptions.shapeEllipse' },
  freehand: { icon: PenLineIcon, labelKey: 'widgets.canvas.toolOptions.shapeFreehand' },
  polygon: { icon: PentagonIcon, labelKey: 'widgets.canvas.toolOptions.shapePolygon' },
  rect: { icon: SquareIcon, labelKey: 'widgets.canvas.toolOptions.shapeRect' },
  star: { icon: StarIcon, labelKey: 'widgets.canvas.toolOptions.shapeStar' },
  triangle: { icon: TriangleIcon, labelKey: 'widgets.canvas.toolOptions.shapeTriangle' },
};

/** The order the kind lists show and the Shape key cycles through. */
export const SHAPE_KIND_ORDER: readonly ShapeToolKind[] = [
  'rect',
  'ellipse',
  'triangle',
  'star',
  'polygon',
  'freehand',
];

export type CanvasToolCommandId = (typeof CANVAS_COMPANION_HOTKEYS)[number][0];

const CANVAS_TOOL_COMMAND_IDS: ReadonlySet<string> = new Set(CANVAS_COMPANION_HOTKEYS.map(([id]) => id));

export const isCanvasToolCommand = (commandId: string): commandId is CanvasToolCommandId =>
  CANVAS_TOOL_COMMAND_IDS.has(commandId);

export type TransformRefusal = Exclude<BeginTransformResult, 'started'>;

/**
 * Explains a refused `beginTransform` on `layer`. A layer Transform cannot act on is a refusal notice (one per layer,
 * not recorded), not a failure; the other refusals keep the shared layer-operation error.
 */
export const reportTransformRefusal = (
  refusal: TransformRefusal,
  layer: { id: string; name: string },
  report: { error: (title: string, message: string) => void; notice: CanvasRefusalNotice },
  t: TFunction
): void => {
  if (refusal === 'ineligible') {
    report.notice(
      `transform:ineligible:${layer.id}`,
      t('widgets.layers.actions.transformUnavailable', { name: layer.name }),
      t('widgets.layers.actions.transformIneligible')
    );
  } else {
    reportLayerOperation(refusal, report.error, t);
  }
};

/** What the companion keys need; the Canvas and the Layers widget each build it from their own state. */
export interface CanvasToolCommandContext {
  readonly engine: Pick<CanvasEngine, 'interaction' | 'layers' | 'tools'> | null;
  /** The selected leaf that Transform opens on; null when nothing or a group is selected. */
  readonly selectedLayerId: string | null;
  /** Swaps the project foreground/background pair — a preference, never a document edit. */
  readonly swapActiveColors: () => void;
  /** Resets the pair to the black-on-white default. */
  readonly resetActiveColors: () => void;
  /** Explains a refused Transform of the selected layer `layerId`. */
  readonly reportTransformRefusal: (refusal: TransformRefusal, layerId: string) => void;
}

/** Runs one companion key. The color pair is a preference, not a document edit: X/D need no engine. */
export const executeCanvasToolCommand = (commandId: CanvasToolCommandId, ctx: CanvasToolCommandContext): void => {
  const { engine } = ctx;
  if (commandId === 'canvas.toggleFillColor') {
    ctx.swapActiveColors();
    return;
  }
  if (commandId === 'canvas.setFillColorsToDefault') {
    ctx.resetActiveColors();
    return;
  }
  if (!engine) {
    return;
  }
  switch (commandId) {
    case 'canvas.tool.view':
      engine.tools.setTool('view');
      return;
    case 'canvas.tool.move':
      engine.tools.setTool('move');
      return;
    case 'canvas.transformSelected': {
      // A selected leaf opens its session at once, even with Transform already active; otherwise only the tool
      // changes. Apply/Cancel (Enter/Escape) are handled engine-side.
      if (ctx.selectedLayerId === null) {
        engine.tools.setTool('transform');
        return;
      }
      const result = engine.layers.beginTransform(ctx.selectedLayerId);
      // A key pressed mid-gesture is dropped quietly; the gesture already says why.
      if (result !== 'started' && result !== 'busy') {
        ctx.reportTransformRefusal(result, ctx.selectedLayerId);
      }
      return;
    }
    case 'canvas.tool.bbox':
      engine.tools.setTool('bbox');
      return;
    case 'canvas.tool.brush':
      engine.tools.setTool('brush');
      return;
    case 'canvas.tool.eraser':
      engine.tools.setTool('eraser');
      return;
    case 'canvas.tool.lasso':
      // Repeating the active tool shortcut cycles its shape.
      if (engine.interaction.get('activeTool') === 'lasso') {
        const lasso = engine.interaction.get('lassoOptions');
        engine.interaction.set('lassoOptions', {
          ...lasso,
          shape: lasso.shape === 'freehand' ? 'polygon' : 'freehand',
        });
      } else {
        engine.tools.setTool('lasso');
      }
      return;
    case 'canvas.tool.marquee':
      if (engine.interaction.get('activeTool') === 'marquee') {
        const marquee = engine.interaction.get('marqueeOptions');
        engine.interaction.set('marqueeOptions', {
          ...marquee,
          kind: marquee.kind === 'rect' ? 'ellipse' : 'rect',
        });
      } else {
        engine.tools.setTool('marquee');
      }
      return;
    case 'canvas.tool.shape':
      if (engine.interaction.get('activeTool') === 'shape') {
        // Repeat presses cycle the kind, like the marquee hotkey.
        const shape = engine.interaction.get('shapeOptions');
        const next = SHAPE_KIND_ORDER[(SHAPE_KIND_ORDER.indexOf(shape.kind) + 1) % SHAPE_KIND_ORDER.length]!;
        engine.interaction.set('shapeOptions', { ...shape, kind: next });
      } else {
        engine.tools.setTool('shape');
      }
      return;
    case 'canvas.tool.text':
      engine.tools.setTool('text');
      return;
    case 'canvas.tool.gradient':
      engine.tools.setTool('gradient');
      return;
    case 'canvas.tool.colorPicker':
      engine.tools.setTool('colorPicker');
      return;
    case 'canvas.brushSizeDown':
      engine.tools.stepBrushSize(-1);
      return;
    case 'canvas.brushSizeUp':
      engine.tools.stepBrushSize(1);
  }
};

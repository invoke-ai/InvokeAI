import type { BeginTransformResult, CanvasEngine } from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';

import { reportLayerOperation } from '@workbench/widgets/canvas/useStructuralCommit';

/** One hotkey row: its command id, title key and default keys. */
export type HotkeyTableEntry = readonly [id: string, titleKey: string, defaultKeys: readonly string[]];

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

export type CanvasToolCommandId = (typeof CANVAS_COMPANION_HOTKEYS)[number][0];

const CANVAS_TOOL_COMMAND_IDS: ReadonlySet<string> = new Set(CANVAS_COMPANION_HOTKEYS.map(([id]) => id));

export const isCanvasToolCommand = (commandId: string): commandId is CanvasToolCommandId =>
  CANVAS_TOOL_COMMAND_IDS.has(commandId);

export type TransformRefusal = Exclude<BeginTransformResult, 'started'>;

/** Explains a refused `beginTransform`; a layer Transform cannot act on gets Transform's own reason. */
export const reportTransformRefusal = (
  refusal: TransformRefusal,
  reportError: (title: string, message: string) => void,
  t: TFunction
): void => {
  if (refusal === 'ineligible') {
    reportError(t('widgets.layers.actions.actionFailed'), t('widgets.layers.actions.transformIneligible'));
  } else {
    reportLayerOperation(refusal, reportError, t);
  }
};

/** What the companion keys need; the Canvas and the Layers widget each build it from their own state. */
export interface CanvasToolCommandContext {
  readonly engine: Pick<CanvasEngine, 'interaction' | 'layers' | 'tools'> | null;
  readonly isInteractionLocked: boolean;
  /** The selected leaf that Transform opens on; null when nothing or a group is selected. */
  readonly selectedLayerId: string | null;
  /** Swaps the project foreground/background pair — a preference, never a document edit. */
  readonly swapActiveColors: () => void;
  /** Resets the pair to the black-on-white default. */
  readonly resetActiveColors: () => void;
  readonly reportTransformRefusal: (refusal: TransformRefusal) => void;
}

/**
 * Runs one companion key. The color pair is a preference, not a document edit, so X/D stay live while the surface
 * is interaction-locked; of the rest only the view tool works then.
 */
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
  if (ctx.isInteractionLocked && commandId !== 'canvas.tool.view') {
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
        ctx.reportTransformRefusal(result);
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
        const order = ['rect', 'ellipse', 'triangle', 'star', 'polygon', 'freehand'] as const;
        const next = order[(order.indexOf(shape.kind) + 1) % order.length]!;
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

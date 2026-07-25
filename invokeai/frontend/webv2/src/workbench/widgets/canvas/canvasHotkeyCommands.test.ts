import type {
  CanvasDocumentContractV2,
  CanvasEngine,
  CanvasInteractionState,
  CanvasLayerContract,
} from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { executeCanvasHotkeyCommand, type CanvasHotkeyContext } from './canvasHotkeyCommands';

const rasterLayer = (id: string, overrides: Partial<CanvasLayerContract> = {}): CanvasLayerContract =>
  ({
    blendMode: 'normal',
    id,
    isEnabled: true,
    isLocked: false,
    name: id,
    opacity: 1,
    source: { bitmap: null, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
    ...overrides,
  }) as CanvasLayerContract;

const documentOf = (layers: readonly CanvasLayerContract[], selectedLayerId: string | null): CanvasDocumentContractV2 =>
  ({
    bbox: { height: 64, width: 64, x: 0, y: 0 },
    height: 64,
    layers,
    selectedLayerId,
    width: 64,
  }) as CanvasDocumentContractV2;

/**
 * A recording engine double. Interaction state is a plain bag so tests can put
 * the engine into "has a pixel selection" or "already on the lasso tool" without
 * standing up the real store.
 */
const createEngine = (interaction: Partial<CanvasInteractionState> = {}) => {
  const state: Record<string, unknown> = {
    activeTool: 'brush',
    hasSelection: false,
    lassoOptions: { shape: 'freehand' },
    marqueeOptions: { kind: 'rect' },
    ...interaction,
  };
  const engine = {
    history: { redo: vi.fn(), undo: vi.fn() },
    interaction: {
      get: vi.fn((key: string) => state[key]),
      set: vi.fn((key: string, value: unknown) => {
        state[key] = value;
      }),
    },
    layers: {
      clearMask: vi.fn(),
      commitStructural: vi.fn(),
      mergeLayerDown: vi.fn(),
      nudgeSelectedLayer: vi.fn(),
    },
    selection: {
      deselect: vi.fn(),
      eraseSelection: vi.fn(),
      invertSelection: vi.fn(),
      liftSelectionToLayer: vi.fn(),
      selectAll: vi.fn(),
    },
    tools: { setTool: vi.fn(), stepBrushSize: vi.fn() },
  };
  return engine as unknown as CanvasEngine & typeof engine;
};

let dispatch: Mock<(mutation: CanvasProjectMutation) => boolean>;
let copySelection: Mock<(cut: boolean) => void>;
let pasteFromClipboard: Mock<() => void>;

const contextOf = (overrides: Partial<CanvasHotkeyContext> = {}): CanvasHotkeyContext => ({
  copySelection,
  createLayerId: () => 'new-layer',
  dispatch,
  document: documentOf([rasterLayer('a'), rasterLayer('b')], 'a'),
  engine: createEngine(),
  hasSelectedStagedCandidate: false,
  hasStagingSlots: false,
  isInteractionLocked: false,
  pasteFromClipboard,
  t: (key) => key,
  ...overrides,
});

const run = (commandId: string, overrides: Partial<CanvasHotkeyContext> = {}) => {
  const ctx = contextOf(overrides);
  executeCanvasHotkeyCommand(commandId, ctx);
  return ctx.engine as ReturnType<typeof createEngine>;
};

beforeEach(() => {
  dispatch = vi.fn();
  copySelection = vi.fn();
  pasteFromClipboard = vi.fn();
});

describe('staging precedence', () => {
  it.each([
    ['canvas.prevEntity', -1],
    ['canvas.nudgeLeft', -1],
    ['canvas.nextEntity', 1],
    ['canvas.nudgeRight', 1],
  ] as const)('%s cycles staged images while slots exist', (commandId, direction) => {
    const engine = run(commandId, { hasStagingSlots: true });
    expect(dispatch).toHaveBeenCalledWith({ direction, type: 'cycleStagedImage' } satisfies CanvasProjectMutation);
    expect(engine.layers.nudgeSelectedLayer).not.toHaveBeenCalled();
  });

  it('nudges instead of cycling when no staging slot exists', () => {
    const engine = run('canvas.nudgeLeft');
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.layers.nudgeSelectedLayer).toHaveBeenCalledWith(-1, 0);
  });

  it('delete discards the selected staged candidate before touching layers', () => {
    const engine = run('canvas.deleteSelected', { hasSelectedStagedCandidate: true });
    expect(dispatch).toHaveBeenCalledWith({ type: 'discardSelectedStagedImage' });
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
    expect(engine.selection.eraseSelection).not.toHaveBeenCalled();
  });

  it('prevEntity does nothing at all without staging slots', () => {
    const engine = run('canvas.prevEntity');
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.tools.setTool).not.toHaveBeenCalled();
  });
});

describe('interaction lock', () => {
  it('allows only the view tool', () => {
    const engine = run('canvas.tool.view', { isInteractionLocked: true });
    expect(engine.tools.setTool).toHaveBeenCalledWith('view');
  });

  it.each([
    'canvas.tool.brush',
    'canvas.deleteSelected',
    'canvas.undo',
    'canvas.selectAll',
    'canvas.nudgeLeft',
    'canvas.layerToFront',
    'canvas.duplicateLayer',
    'canvas.mergeDown',
  ])('blocks %s', (commandId) => {
    const engine = run(commandId, { isInteractionLocked: true });
    expect(engine.tools.setTool).not.toHaveBeenCalled();
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
    expect(engine.layers.nudgeSelectedLayer).not.toHaveBeenCalled();
    expect(engine.history.undo).not.toHaveBeenCalled();
    expect(engine.selection.selectAll).not.toHaveBeenCalled();
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('still cycles staging while locked (staging is checked first)', () => {
    run('canvas.nextEntity', { hasStagingSlots: true, isInteractionLocked: true });
    expect(dispatch).toHaveBeenCalledWith({ direction: 1, type: 'cycleStagedImage' });
  });
});

describe('nudge', () => {
  it.each([
    ['canvas.nudgeUp', 0, -1],
    ['canvas.nudgeUpLarge', 0, -10],
    ['canvas.nudgeDown', 0, 1],
    ['canvas.nudgeDownLarge', 0, 10],
    ['canvas.nudgeLeftLarge', -10, 0],
    ['canvas.nudgeRightLarge', 10, 0],
  ])('%s nudges by (%i, %i)', (commandId, dx, dy) => {
    const engine = run(commandId);
    expect(engine.layers.nudgeSelectedLayer).toHaveBeenCalledWith(dx, dy);
  });

  it('large nudges are unaffected by staging slots', () => {
    const engine = run('canvas.nudgeRightLarge', { hasStagingSlots: true });
    expect(engine.layers.nudgeSelectedLayer).toHaveBeenCalledWith(10, 0);
    expect(dispatch).not.toHaveBeenCalled();
  });
});

describe('layer reorder', () => {
  it.each(['canvas.layerForward', 'canvas.layerBackward', 'canvas.layerToFront', 'canvas.layerToBack'])(
    '%s commits through the engine history',
    (commandId) => {
      const engine = run(commandId, {
        document: documentOf([rasterLayer('a'), rasterLayer('b'), rasterLayer('c')], 'b'),
      });
      expect(engine.layers.commitStructural).toHaveBeenCalledWith(
        'widgets.canvas.commands.reorderLayer',
        expect.anything(),
        expect.anything()
      );
    }
  );

  it('is a no-op at the edge of the stack', () => {
    const engine = run('canvas.layerToFront', { document: documentOf([rasterLayer('a'), rasterLayer('b')], 'a') });
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
  });

  it('is a no-op with no selected layer', () => {
    const engine = run('canvas.layerBackward', { document: documentOf([rasterLayer('a')], null) });
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
  });

  it('is a no-op without an engine', () => {
    expect(() => executeCanvasHotkeyCommand('canvas.layerForward', contextOf({ engine: null }))).not.toThrow();
  });
});

describe('delete: pixels vs layer', () => {
  it('erases the pixel selection when one exists', () => {
    const engine = run('canvas.deleteSelected', { engine: createEngine({ hasSelection: true }) });
    expect(engine.selection.eraseSelection).toHaveBeenCalled();
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
  });

  it('deletes the selected layer when there is no pixel selection', () => {
    const engine = run('canvas.deleteSelected');
    expect(engine.selection.eraseSelection).not.toHaveBeenCalled();
    expect(engine.layers.commitStructural).toHaveBeenCalledWith(
      'widgets.canvas.commands.deleteLayer',
      expect.anything(),
      expect.anything()
    );
  });

  it('does nothing with neither a selection nor a selected layer', () => {
    const engine = run('canvas.deleteSelected', { document: documentOf([rasterLayer('a')], null) });
    expect(engine.selection.eraseSelection).not.toHaveBeenCalled();
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
  });
});

describe('duplicate: lift vs whole layer', () => {
  it('lifts the pixel selection into a new layer when one exists', () => {
    const engine = run('canvas.duplicateLayer', { engine: createEngine({ hasSelection: true }) });
    expect(engine.selection.liftSelectionToLayer).toHaveBeenCalled();
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
  });

  it('duplicates the whole layer with no pixel selection, using the injected id', () => {
    const engine = run('canvas.duplicateLayer');
    expect(engine.layers.commitStructural).toHaveBeenCalledWith(
      'widgets.canvas.commands.duplicateLayer',
      { newId: 'new-layer', sourceId: 'a', type: 'duplicateCanvasLayer' },
      { ids: ['new-layer'], type: 'removeCanvasLayers' }
    );
  });
});

describe('clipboard', () => {
  it('copies without cutting', () => {
    run('canvas.copySelection');
    expect(copySelection).toHaveBeenCalledWith(false);
  });

  it('cuts', () => {
    run('canvas.cutSelection');
    expect(copySelection).toHaveBeenCalledWith(true);
  });

  it('pastes', () => {
    run('canvas.pasteImage');
    expect(pasteFromClipboard).toHaveBeenCalled();
  });
});

describe('tool selection', () => {
  it.each([
    ['canvas.tool.view', 'view'],
    ['canvas.tool.move', 'move'],
    ['canvas.tool.bbox', 'bbox'],
    ['canvas.tool.brush', 'brush'],
    ['canvas.tool.eraser', 'eraser'],
    ['canvas.tool.shape', 'shape'],
    ['canvas.tool.text', 'text'],
    ['canvas.tool.gradient', 'gradient'],
    ['canvas.transformSelected', 'transform'],
  ])('%s selects the %s tool', (commandId, tool) => {
    const engine = run(commandId);
    expect(engine.tools.setTool).toHaveBeenCalledWith(tool);
  });

  it('lasso selects the tool when not already active', () => {
    const engine = run('canvas.tool.lasso');
    expect(engine.tools.setTool).toHaveBeenCalledWith('lasso');
    expect(engine.interaction.set).not.toHaveBeenCalled();
  });

  it('lasso cycles freehand → polygon when already active', () => {
    const engine = run('canvas.tool.lasso', {
      engine: createEngine({ activeTool: 'lasso', lassoOptions: { shape: 'freehand' } as never }),
    });
    expect(engine.tools.setTool).not.toHaveBeenCalled();
    expect(engine.interaction.set).toHaveBeenCalledWith('lassoOptions', { shape: 'polygon' });
  });

  it('lasso cycles polygon → freehand', () => {
    const engine = run('canvas.tool.lasso', {
      engine: createEngine({ activeTool: 'lasso', lassoOptions: { shape: 'polygon' } as never }),
    });
    expect(engine.interaction.set).toHaveBeenCalledWith('lassoOptions', { shape: 'freehand' });
  });

  it('marquee cycles rect → ellipse when already active', () => {
    const engine = run('canvas.tool.marquee', {
      engine: createEngine({ activeTool: 'marquee', marqueeOptions: { kind: 'rect' } as never }),
    });
    expect(engine.tools.setTool).not.toHaveBeenCalled();
    expect(engine.interaction.set).toHaveBeenCalledWith('marqueeOptions', { kind: 'ellipse' });
  });

  it('marquee cycles ellipse → rect', () => {
    const engine = run('canvas.tool.marquee', {
      engine: createEngine({ activeTool: 'marquee', marqueeOptions: { kind: 'ellipse' } as never }),
    });
    expect(engine.interaction.set).toHaveBeenCalledWith('marqueeOptions', { kind: 'rect' });
  });
});

describe('history, selection, and brush size', () => {
  it('undo and redo drive the engine history, never the reducer', () => {
    expect(run('canvas.undo').history.undo).toHaveBeenCalled();
    expect(run('canvas.redo').history.redo).toHaveBeenCalled();
    expect(dispatch).not.toHaveBeenCalled();
  });

  it.each([
    ['canvas.selectAll', 'selectAll'],
    ['canvas.deselect', 'deselect'],
    ['canvas.invertSelection', 'invertSelection'],
  ] as const)('%s calls selection.%s', (commandId, method) => {
    expect(run(commandId).selection[method]).toHaveBeenCalled();
  });

  it.each([
    ['canvas.brushSizeDown', -1],
    ['canvas.brushSizeUp', 1],
  ])('%s steps the brush size by %i', (commandId, step) => {
    expect(run(commandId).tools.stepBrushSize).toHaveBeenCalledWith(step);
  });

  it('resetSelected clears the selected layer mask', () => {
    expect(run('canvas.resetSelected').layers.clearMask).toHaveBeenCalledWith('a');
  });
});

describe('mergeDown', () => {
  it('merges two mergeable rasters', () => {
    const engine = run('canvas.mergeDown');
    expect(engine.layers.mergeLayerDown).toHaveBeenCalledWith('a');
  });

  it('refuses when the selected layer is bottom-most', () => {
    const engine = run('canvas.mergeDown', { document: documentOf([rasterLayer('a'), rasterLayer('b')], 'b') });
    expect(engine.layers.mergeLayerDown).not.toHaveBeenCalled();
  });

  it('refuses with no selected layer', () => {
    const engine = run('canvas.mergeDown', { document: documentOf([rasterLayer('a')], null) });
    expect(engine.layers.mergeLayerDown).not.toHaveBeenCalled();
  });
});

describe('toggleNonRasterLayers', () => {
  it('does nothing when there are no hideable layers', () => {
    const engine = run('canvas.toggleNonRasterLayers');
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
  });

  it('hides all hideable layers when all are visible, and restores prior state on undo', () => {
    const mask = rasterLayer('m', { type: 'inpaint_mask' } as Partial<CanvasLayerContract>);
    const engine = run('canvas.toggleNonRasterLayers', { document: documentOf([rasterLayer('a'), mask], 'a') });
    expect(engine.layers.commitStructural).toHaveBeenCalledWith(
      'widgets.canvas.commands.toggleNonRasterLayers',
      { type: 'setCanvasLayersHidden', updates: [{ id: 'm', isHidden: true }] },
      { type: 'setCanvasLayersHidden', updates: [{ id: 'm', isHidden: false }] }
    );
  });

  it('reveals hideable layers when any is already hidden', () => {
    const mask = rasterLayer('m', { isHidden: true, type: 'inpaint_mask' } as Partial<CanvasLayerContract>);
    const engine = run('canvas.toggleNonRasterLayers', { document: documentOf([rasterLayer('a'), mask], 'a') });
    expect(engine.layers.commitStructural).toHaveBeenCalledWith(
      'widgets.canvas.commands.toggleNonRasterLayers',
      { type: 'setCanvasLayersHidden', updates: [{ id: 'm', isHidden: false }] },
      { type: 'setCanvasLayersHidden', updates: [{ id: 'm', isHidden: true }] }
    );
  });
});

describe('robustness', () => {
  it('ignores an unknown command id', () => {
    const engine = run('canvas.nonexistent');
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.tools.setTool).not.toHaveBeenCalled();
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
  });

  it('never throws without an engine', () => {
    for (const commandId of [
      'canvas.undo',
      'canvas.redo',
      'canvas.deleteSelected',
      'canvas.duplicateLayer',
      'canvas.mergeDown',
      'canvas.selectAll',
      'canvas.tool.lasso',
      'canvas.tool.marquee',
      'canvas.toggleNonRasterLayers',
      'canvas.resetSelected',
      'canvas.nudgeUp',
      'canvas.brushSizeUp',
    ]) {
      expect(() => executeCanvasHotkeyCommand(commandId, contextOf({ engine: null }))).not.toThrow();
    }
  });
});

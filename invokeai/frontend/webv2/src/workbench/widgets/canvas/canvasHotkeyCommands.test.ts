import type {
  CanvasDocumentContractV3,
  CanvasEngine,
  CanvasInteractionState,
  CanvasLayerContract,
  CanvasNodeContract,
  MergeDownResult,
  NewRasterLayerResult,
} from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { createDocumentModel } from '@workbench/canvas-engine/api';
import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createControlLayer } from '@workbench/widgets/layers/layerOps';
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

const documentOf = (layers: readonly CanvasLayerContract[], selectedLayerId: string | null): CanvasDocumentContractV3 =>
  ({
    bbox: { height: 64, width: 64, x: 0, y: 0 },
    height: 64,
    stacks: stacksFrom(layers),
    selectedLayerId,
    width: 64,
  }) as CanvasDocumentContractV3;

/** Use a recording engine with mutable interaction state for selection/tool scenarios without real stores. */
const createEngine = (interaction: Partial<CanvasInteractionState> = {}) => {
  const state: Record<string, unknown> = {
    activeTool: 'brush',
    hasSelection: false,
    lassoOptions: { shape: 'freehand' },
    marqueeOptions: { kind: 'rect' },
    ...interaction,
  };
  const engine = {
    projectId: 'project',
    document: {
      captureRestoreAnchor: () => stackTopAnchor('project'),
      model: vi.fn(),
    },
    history: { redo: vi.fn(), undo: vi.fn() },
    interaction: {
      get: vi.fn((key: string) => state[key]),
      set: vi.fn((key: string, value: unknown) => {
        state[key] = value;
      }),
    },
    layers: {
      clearMask: vi.fn(() => ({ status: 'committed' })),
      commitPrepared: vi.fn(() => ({ status: 'committed' as const })),
      commitStructural: vi.fn(),
      duplicateLayers: vi.fn(() =>
        Promise.resolve<
          | { status: 'duplicated'; duplicateIds: string[]; selectedLayerId: string }
          | { status: 'not-ready' | 'over-budget' | 'stale' }
        >({
          duplicateIds: ['new-layer'],
          selectedLayerId: 'new-layer',
          status: 'duplicated',
        })
      ),
      mergeLayerDown: vi.fn((): MergeDownResult => 'merged'),
      nudgeSelectedLayer: vi.fn(() => ({ status: 'committed' as const })),
    },
    selection: {
      deselect: vi.fn(),
      eraseSelection: vi.fn(),
      invertSelection: vi.fn(),
      liftSelectionToLayer: vi.fn((): NewRasterLayerResult => ({ layerId: 'lifted', status: 'created' })),
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
  dispatch,
  reportLayerOperation: vi.fn(),
  reportPreparedCommit: () => undefined,
  reportStructuralCommit: () => undefined,
  document: documentOf([rasterLayer('a'), rasterLayer('b')], 'a'),
  engine: createEngine(),
  hasSelectedStagedCandidate: false,
  hasStagingSlots: false,
  isInteractionLocked: false,
  notifyLayerDuplicateFailed: vi.fn(),
  pasteFromClipboard,
  resetActiveColors: vi.fn(),
  selectedLayerIds: ['a'],
  swapActiveColors: vi.fn(),
  t: (key) => key,
  ...overrides,
});

const run = (commandId: string, overrides: Partial<CanvasHotkeyContext> = {}) => {
  const ctx = contextOf(overrides);
  (ctx.engine!.document.model as Mock).mockImplementation(() =>
    createDocumentModel(ctx.document, { editRevision: 0, projectId: 'project' })
  );
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
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
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
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
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
        selectedLayerIds: ['b'],
      });
      expect(engine.layers.commitPrepared).toHaveBeenCalledWith(
        'widgets.canvas.commands.reorderLayer',
        expect.objectContaining({ forward: expect.objectContaining({ type: 'reorderCanvasSiblings' }) })
      );
    }
  );

  it('is a no-op at the edge of the stack', () => {
    const engine = run('canvas.layerToFront', { document: documentOf([rasterLayer('a'), rasterLayer('b')], 'a') });
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
  });

  it('is a no-op with no selected layer', () => {
    const engine = run('canvas.layerBackward', { document: documentOf([rasterLayer('a')], null) });
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
  });

  it('is a no-op without an engine', () => {
    expect(() => executeCanvasHotkeyCommand('canvas.layerForward', contextOf({ engine: null }))).not.toThrow();
  });

  it('moves the full multi-selection within each layer type group', () => {
    const layers = [
      rasterLayer('c1', { type: 'control' }),
      rasterLayer('r1'),
      rasterLayer('c2', { type: 'control' }),
      rasterLayer('r2'),
    ];
    const engine = run('canvas.layerForward', {
      document: documentOf(layers, 'c2'),
      selectedLayerIds: ['c2', 'r2'],
    });

    expect(engine.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.reorderLayer',
      expect.objectContaining({
        forward: {
          orders: [
            { orderedIds: ['c2', 'c1'], parentId: null, stack: 'control' },
            { orderedIds: ['r2', 'r1'], parentId: null, stack: 'raster' },
          ],
          type: 'reorderCanvasSiblings',
        },
      })
    );
  });
});

describe('delete: pixels vs layer', () => {
  it('erases the pixel selection when one exists', () => {
    const engine = run('canvas.deleteSelected', { engine: createEngine({ hasSelection: true }) });
    expect(engine.selection.eraseSelection).toHaveBeenCalled();
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
  });

  it('deletes the selected layer when there is no pixel selection', () => {
    const engine = run('canvas.deleteSelected');
    expect(engine.selection.eraseSelection).not.toHaveBeenCalled();
    expect(engine.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.deleteLayer',
      expect.objectContaining({ forward: { ids: ['a'], type: 'removeCanvasLayers' } })
    );
  });

  it('deletes the full Layers-panel selection as one history edit', () => {
    const layers = [rasterLayer('a'), rasterLayer('b'), rasterLayer('c')];
    const engine = run('canvas.deleteSelected', { document: documentOf(layers, 'a'), selectedLayerIds: ['a', 'c'] });

    expect(engine.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.deleteLayer',
      expect.objectContaining({
        forward: { ids: ['a', 'c'], type: 'removeCanvasLayers' },
        inverse: expect.objectContaining({
          add: [expect.objectContaining({ nodes: [layers[0]] }), expect.objectContaining({ nodes: [layers[2]] })],
          selectedLayerId: 'a',
        }),
      })
    );
  });

  it('does not delete when any selected layer is locked', () => {
    const engine = run('canvas.deleteSelected', {
      document: documentOf([rasterLayer('a'), rasterLayer('b', { isLocked: true })], 'a'),
      selectedLayerIds: ['a', 'b'],
    });

    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
  });

  it('does nothing with neither a selection nor a selected layer', () => {
    const engine = run('canvas.deleteSelected', { document: documentOf([rasterLayer('a')], null) });
    expect(engine.selection.eraseSelection).not.toHaveBeenCalled();
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
  });
});

describe('duplicate: lift vs whole layer', () => {
  it('lifts the pixel selection into a new layer when one exists', () => {
    const engine = run('canvas.duplicateLayer', { engine: createEngine({ hasSelection: true }) });
    expect(engine.selection.liftSelectionToLayer).toHaveBeenCalled();
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
  });

  it('duplicates the whole layer with no pixel selection', () => {
    const engine = run('canvas.duplicateLayer');
    expect(engine.layers.duplicateLayers).toHaveBeenCalledWith(['a']);
  });

  it('duplicates every selected layer as one engine operation', () => {
    const engine = run('canvas.duplicateLayer', { selectedLayerIds: ['a', 'b'] });
    expect(engine.layers.duplicateLayers).toHaveBeenCalledWith(['a', 'b']);
  });

  it('reports when whole-layer duplication throws', async () => {
    const engine = createEngine();
    engine.layers.duplicateLayers.mockImplementation(() => Promise.reject(new Error('rejected')));
    const notifyLayerDuplicateFailed = vi.fn();

    executeCanvasHotkeyCommand('canvas.duplicateLayer', contextOf({ engine, notifyLayerDuplicateFailed }));

    await vi.waitFor(() => expect(notifyLayerDuplicateFailed).toHaveBeenCalledOnce());
  });

  it.each(['not-ready', 'over-budget', 'stale'] as const)(
    'explains a %s whole-layer duplication refusal',
    async (status) => {
      const engine = createEngine();
      engine.layers.duplicateLayers.mockImplementation(() => Promise.resolve({ status }));
      const reportLayerOperation = vi.fn();

      executeCanvasHotkeyCommand('canvas.duplicateLayer', contextOf({ engine, reportLayerOperation }));

      await vi.waitFor(() => expect(reportLayerOperation).toHaveBeenCalledExactlyOnceWith(status));
    }
  );
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

  it('shape selects the tool when it is not active', () => {
    const engine = run('canvas.tool.shape', {
      engine: createEngine({ shapeOptions: { kind: 'triangle' } as never }),
    });
    expect(engine.tools.setTool).toHaveBeenCalledWith('shape');
    expect(engine.interaction.set).not.toHaveBeenCalledWith('shapeOptions', expect.anything());
  });

  it('shape cycles rect → ellipse when already active, keeping the other options', () => {
    const engine = run('canvas.tool.shape', {
      engine: createEngine({ activeTool: 'shape', shapeOptions: { fillEnabled: true, kind: 'rect' } as never }),
    });
    expect(engine.tools.setTool).not.toHaveBeenCalled();
    expect(engine.interaction.set).toHaveBeenCalledWith('shapeOptions', { fillEnabled: true, kind: 'ellipse' });
  });

  it('shape cycles star on to polygon, and freehand back around to rect', () => {
    const fromStar = run('canvas.tool.shape', {
      engine: createEngine({ activeTool: 'shape', shapeOptions: { kind: 'star' } as never }),
    });
    expect(fromStar.interaction.set).toHaveBeenCalledWith('shapeOptions', { kind: 'polygon' });
    const fromFreehand = run('canvas.tool.shape', {
      engine: createEngine({ activeTool: 'shape', shapeOptions: { kind: 'freehand' } as never }),
    });
    expect(fromFreehand.interaction.set).toHaveBeenCalledWith('shapeOptions', { kind: 'rect' });
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

  it('resetSelected explains a refused clear and stays silent when there is nothing to clear', () => {
    const engine = createEngine();
    const reportLayerOperation = vi.fn();
    engine.layers.clearMask.mockReturnValueOnce({ status: 'over-budget' }).mockReturnValueOnce({ status: 'nothing' });

    executeCanvasHotkeyCommand('canvas.resetSelected', contextOf({ engine, reportLayerOperation }));
    executeCanvasHotkeyCommand('canvas.resetSelected', contextOf({ engine, reportLayerOperation }));

    expect(reportLayerOperation).toHaveBeenCalledExactlyOnceWith('over-budget');
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

  it('reports a merge the engine refused', () => {
    const engine = createEngine();
    engine.layers.mergeLayerDown.mockReturnValue('over-budget');
    const reportLayerOperation = vi.fn();
    executeCanvasHotkeyCommand('canvas.mergeDown', contextOf({ engine, reportLayerOperation }));
    expect(reportLayerOperation).toHaveBeenCalledExactlyOnceWith('over-budget');
  });
});

describe('toggleNonRasterLayers', () => {
  it('does nothing when there are no hideable layers', () => {
    const engine = run('canvas.toggleNonRasterLayers');
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
  });

  it('hides all hideable layers when all are visible, and restores prior state on undo', () => {
    const mask = rasterLayer('m', { type: 'inpaint_mask' } as Partial<CanvasLayerContract>);
    const engine = run('canvas.toggleNonRasterLayers', { document: documentOf([rasterLayer('a'), mask], 'a') });
    expect(engine.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.toggleNonRasterLayers',
      expect.objectContaining({
        forward: { type: 'setCanvasLayersHidden', updates: [{ id: 'm', isHidden: true }] },
        inverse: { type: 'setCanvasLayersHidden', updates: [{ id: 'm', isHidden: false }] },
      })
    );
  });

  it('reveals hideable layers when any is already hidden', () => {
    const mask = rasterLayer('m', { isHidden: true, type: 'inpaint_mask' } as Partial<CanvasLayerContract>);
    const engine = run('canvas.toggleNonRasterLayers', { document: documentOf([rasterLayer('a'), mask], 'a') });
    expect(engine.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.toggleNonRasterLayers',
      expect.objectContaining({
        forward: { type: 'setCanvasLayersHidden', updates: [{ id: 'm', isHidden: false }] },
        inverse: { type: 'setCanvasLayersHidden', updates: [{ id: 'm', isHidden: true }] },
      })
    );
  });
});

describe('robustness', () => {
  it('ignores an unknown command id', () => {
    const engine = run('canvas.nonexistent');
    expect(dispatch).not.toHaveBeenCalled();
    expect(engine.tools.setTool).not.toHaveBeenCalled();
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
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

describe('group selection', () => {
  const nodeDocument = (nodes: CanvasNodeContract[], selectedLayerId: string | null): CanvasDocumentContractV3 => ({
    ...documentOf([], selectedLayerId),
    stacks: stacksFrom(nodes),
  });
  const control = (id: string): CanvasLayerContract => createControlLayer(id, id);

  it('reorders, deletes and duplicates a selected group', () => {
    const document = nodeDocument([groupContract('g', [rasterLayer('a')]), rasterLayer('b')], 'g');
    const reorder = run('canvas.layerToBack', { document, selectedLayerIds: ['g'] });
    expect(reorder.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.reorderLayer',
      expect.objectContaining({ forward: expect.objectContaining({ type: 'reorderCanvasSiblings' }) })
    );
    const remove = run('canvas.deleteSelected', { document, selectedLayerIds: ['g'] });
    expect(remove.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.deleteLayer',
      expect.objectContaining({ forward: { ids: ['g'], type: 'removeCanvasLayers' } })
    );
    const duplicate = run('canvas.duplicateLayer', { document, selectedLayerIds: ['g'] });
    expect(duplicate.layers.duplicateLayers).toHaveBeenCalledWith(['g']);
  });

  it('refuses to delete a leaf frozen by a locked group', () => {
    const document = nodeDocument([groupContract('g', [rasterLayer('a')], { isLocked: true })], 'a');
    const engine = run('canvas.deleteSelected', { document, selectedLayerIds: ['a'] });
    expect(engine.layers.commitPrepared).not.toHaveBeenCalled();
  });

  it('shows overlays hidden behind a hidden group, and hides them again from the roots', () => {
    const hidden = nodeDocument([{ ...groupContract('g', [control('c1')]), isHidden: true }, control('c2')], 'c2');
    const shown = run('canvas.toggleNonRasterLayers', { document: hidden });
    expect(shown.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.toggleNonRasterLayers',
      expect.objectContaining({ forward: { type: 'setCanvasLayersHidden', updates: [{ id: 'g', isHidden: false }] } })
    );
    const visible = nodeDocument([groupContract('g', [control('c1')]), control('c2')], 'c2');
    const hide = run('canvas.toggleNonRasterLayers', { document: visible });
    expect(hide.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.toggleNonRasterLayers',
      expect.objectContaining({
        forward: {
          type: 'setCanvasLayersHidden',
          updates: [
            { id: 'g', isHidden: true },
            { id: 'c2', isHidden: true },
          ],
        },
      })
    );
  });

  it('groups and ungroups the selection', () => {
    const grouped = run('canvas.groupLayers', {
      document: nodeDocument([rasterLayer('a'), rasterLayer('b')], 'a'),
      selectedLayerIds: ['a', 'b'],
    });
    expect(grouped.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.groupLayers',
      expect.objectContaining({
        forward: expect.objectContaining({
          add: [expect.objectContaining({ nodes: [expect.objectContaining({ type: 'group' })] })],
        }),
      })
    );
    const ungrouped = run('canvas.ungroupLayers', {
      document: nodeDocument([groupContract('g', [rasterLayer('a')])], 'g'),
      selectedLayerIds: ['g'],
    });
    expect(ungrouped.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.ungroupLayers',
      expect.objectContaining({ forward: expect.objectContaining({ removeIds: ['g'] }) })
    );
  });
});

describe('active color pair commands', () => {
  it('routes X to swap and D to reset, even while interaction is locked', () => {
    for (const isInteractionLocked of [false, true]) {
      const swapCtx = contextOf({ isInteractionLocked });
      executeCanvasHotkeyCommand('canvas.toggleFillColor', swapCtx);
      expect(swapCtx.swapActiveColors).toHaveBeenCalledTimes(1);
      expect(swapCtx.resetActiveColors).not.toHaveBeenCalled();

      const resetCtx = contextOf({ isInteractionLocked });
      executeCanvasHotkeyCommand('canvas.setFillColorsToDefault', resetCtx);
      expect(resetCtx.resetActiveColors).toHaveBeenCalledTimes(1);
      expect(resetCtx.dispatch).not.toHaveBeenCalled();
    }
  });
});

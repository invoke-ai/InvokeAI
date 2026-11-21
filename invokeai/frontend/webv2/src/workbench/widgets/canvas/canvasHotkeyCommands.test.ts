import type {
  BeginTransformResult,
  CanvasDocumentContractV3,
  CanvasEngine,
  CanvasInteractionState,
  CanvasLayerContract,
  CanvasNodeContract,
  MergeDownResult,
  NewRasterLayerResult,
} from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type * as LayerCommandsModule from '@workbench/widgets/layers/layerCommands';
import type { TFunction } from 'i18next';

import { createControlLayer, createDocumentModel } from '@workbench/canvas-engine/api';
import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { deleteLayers, duplicateLayers, groupLayers, ungroupLayers } from '@workbench/widgets/layers/layerCommands';
import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

// The layer commands own their eligibility, labels and feedback (see layerCommands tests); here only routing counts.
vi.mock('@workbench/widgets/layers/layerCommands', async (importOriginal) => ({
  ...(await importOriginal<typeof LayerCommandsModule>()),
  deleteLayers: vi.fn(() => 'committed'),
  duplicateLayers: vi.fn(() => Promise.resolve('committed')),
  groupLayers: vi.fn(() => 'committed'),
  ungroupLayers: vi.fn(() => 'committed'),
}));

import { CANVAS_HEADER_HOTKEYS } from './canvasHeaderCommands';
import { CANVAS_ONLY_HOTKEYS, executeCanvasHotkeyCommand, type CanvasHotkeyContext } from './canvasHotkeyCommands';
import { CANVAS_COMPANION_HOTKEYS, reportMergeDownRefusal } from './canvasToolCommands';

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

const createEngine = (interaction: Partial<CanvasInteractionState> = {}) => {
  const state: Record<string, unknown> = {
    activeTool: 'brush',
    hasSelection: false,
    lassoOptions: { shape: 'freehand' },
    layerQuickSwitch: { bookmarkedLayerId: null, previousLayerId: null },
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
      endStructuralPreview: vi.fn(),
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
      beginTransform: vi.fn((): BeginTransformResult => 'started'),
      mergeLayerDown: vi.fn((): Promise<MergeDownResult> => Promise.resolve('merged')),
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
  cropAllLayers: vi.fn(),
  dispatch,
  reportLayerOperation: vi.fn(),
  reportMergeDownRefusal: vi.fn(),
  reportTransformRefusal: vi.fn(),
  reportPreparedCommit: () => undefined,
  reportStructuralCommit: () => undefined,
  document: documentOf([rasterLayer('a'), rasterLayer('b')], 'a'),
  engine: createEngine(),
  hasMultipleStagingSlots: false,
  layerCommands: { engine: null, reportError: vi.fn(), t: ((key: string) => key) as TFunction },
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
  vi.clearAllMocks();
  dispatch = vi.fn();
  copySelection = vi.fn();
  pasteFromClipboard = vi.fn();
});

describe('while results are staged', () => {
  it.each([
    ['canvas.prevEntity', -1],
    ['canvas.nextEntity', 1],
  ] as const)('%s steps through the staged results, and only that', (commandId, direction) => {
    const engine = run(commandId, { hasMultipleStagingSlots: true });
    expect(dispatch).toHaveBeenCalledExactlyOnceWith({
      direction,
      type: 'cycleStagedImage',
    } satisfies CanvasProjectMutation);
    expect(engine.layers.nudgeSelectedLayer).not.toHaveBeenCalled();
  });

  it('Delete erases the pixel selection, or deletes the layer, instead of discarding a result', () => {
    const erasing = run('canvas.deleteSelected', {
      engine: createEngine({ hasSelection: true }),
      hasMultipleStagingSlots: true,
    });
    expect(erasing.selection.eraseSelection).toHaveBeenCalledOnce();
    expect(dispatch).not.toHaveBeenCalled();

    run('canvas.deleteSelected', { hasMultipleStagingSlots: true });
    expect(deleteLayers).toHaveBeenCalledOnce();
    expect(dispatch).not.toHaveBeenCalled();
  });
});

describe('layer navigation without staging', () => {
  const select = (id: string): CanvasProjectMutation => ({ id, type: 'setCanvasSelectedLayer' });
  // Document order puts the control stack above the raster stack: b, a, c.
  const layers = [rasterLayer('a'), createControlLayer({ id: 'b', name: 'Control' }), rasterLayer('c')];

  it.each([
    ['canvas.nextEntity', 'a', 'c'],
    ['canvas.nextEntity', 'c', 'b'],
    ['canvas.prevEntity', 'b', 'c'],
    ['canvas.prevEntity', 'a', 'b'],
  ] as const)('%s from %s selects %s, wrapping across every layer kind in layer order', (commandId, from, to) => {
    const engine = run(commandId, { document: documentOf(layers, from), selectedLayerIds: [from] });
    expect(dispatch).toHaveBeenCalledExactlyOnceWith(select(to));
    expect(engine.layers.nudgeSelectedLayer).not.toHaveBeenCalled();
  });

  it('quick switch prefers the bookmark, falls back to the previous selection, and skips missing layers', () => {
    const quickSwitch = (bookmarkedLayerId: string | null, previousLayerId: string | null, selected: string) =>
      run('canvas.quickSwitch', {
        document: documentOf(layers, selected),
        engine: createEngine({ layerQuickSwitch: { bookmarkedLayerId, previousLayerId } }),
        selectedLayerIds: [selected],
      });

    quickSwitch('c', 'b', 'a');
    expect(dispatch).toHaveBeenLastCalledWith(select('c'));
    quickSwitch('c', 'a', 'c');
    expect(dispatch).toHaveBeenLastCalledWith(select('a'));
    quickSwitch('gone', 'b', 'a');
    expect(dispatch).toHaveBeenLastCalledWith(select('b'));
    dispatch.mockClear();
    quickSwitch(null, null, 'a');
    quickSwitch(null, 'a', 'a');
    quickSwitch('gone', 'gone', 'a');
    executeCanvasHotkeyCommand('canvas.quickSwitch', contextOf({ engine: null }));
    expect(dispatch).not.toHaveBeenCalled();
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
});

describe('layer commands', () => {
  it.each([
    ['canvas.deleteSelected', deleteLayers],
    ['canvas.duplicateLayer', duplicateLayers],
    ['canvas.groupLayers', groupLayers],
    ['canvas.ungroupLayers', ungroupLayers],
  ])('%s runs the shared layer command on the Layers-panel selection', (commandId, command) => {
    const ctx = contextOf({ selectedLayerIds: ['a', 'b'] });
    executeCanvasHotkeyCommand(commandId, ctx);
    expect(command).toHaveBeenCalledExactlyOnceWith(ctx.layerCommands, ['a', 'b']);
  });

  it('Delete erases a live pixel selection instead of deleting layers', () => {
    const engine = run('canvas.deleteSelected', { engine: createEngine({ hasSelection: true }) });
    expect(engine.selection.eraseSelection).toHaveBeenCalledOnce();
    expect(deleteLayers).not.toHaveBeenCalled();
  });

  it('mod+J lifts a live pixel selection into a new layer instead of duplicating layers', () => {
    const engine = run('canvas.duplicateLayer', { engine: createEngine({ hasSelection: true }) });
    expect(engine.selection.liftSelectionToLayer).toHaveBeenCalledOnce();
    expect(duplicateLayers).not.toHaveBeenCalled();
  });
});

describe('clipboard', () => {
  it.each([
    ['canvas.copySelection', false],
    ['canvas.cutSelection', true],
  ] as const)('%s copies the selection with cut %s', (commandId, cut) => {
    run(commandId);
    expect(copySelection).toHaveBeenCalledExactlyOnceWith(cut);
  });

  it('pastes', () => {
    run('canvas.pasteImage');
    expect(pasteFromClipboard).toHaveBeenCalled();
  });
});

describe('crop all layers to frame', () => {
  it('runs the canvas menu action', () => {
    const ctx = contextOf();
    executeCanvasHotkeyCommand('canvas.cropAllLayersToFrame', ctx);
    expect(ctx.cropAllLayers).toHaveBeenCalledOnce();
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
  ])('%s selects the %s tool', (commandId, tool) => {
    const engine = run(commandId);
    expect(engine.tools.setTool).toHaveBeenCalledWith(tool);
  });

  it('transform opens a session on the selected leaf at once, without a separate tool switch', () => {
    const engine = run('canvas.transformSelected');
    expect(engine.layers.beginTransform).toHaveBeenCalledExactlyOnceWith('a');
    expect(engine.tools.setTool).not.toHaveBeenCalled();
  });
});

describe('history, selection, and brush size', () => {
  it('undo and redo drive the engine history, never the reducer', () => {
    expect(run('canvas.undo').history.undo).toHaveBeenCalled();
    expect(run('canvas.redo').history.redo).toHaveBeenCalled();
    expect(dispatch).not.toHaveBeenCalled();
  });
});

describe('mergeDown', () => {
  it('merges two mergeable rasters', () => {
    const engine = run('canvas.mergeDown');
    expect(engine.layers.mergeLayerDown).toHaveBeenCalledWith('a');
  });

  const explainMergeDown = (document: CanvasDocumentContractV3): unknown[][] => {
    const info = vi.fn();
    const t = ((key: string, options?: { name?: string }) =>
      options?.name === undefined ? key : `${key}:${options.name}`) as unknown as TFunction;
    const engine = run('canvas.mergeDown', {
      document,
      reportMergeDownRefusal: (refusal) => reportMergeDownRefusal(document, refusal, info, t),
    });
    expect(engine.layers.mergeLayerDown).not.toHaveBeenCalled();
    return info.mock.calls;
  };
  const title = 'widgets.layers.actions.mergeDownRefused:b';
  const shape = (id: string, overrides: Partial<CanvasLayerContract> = {}) =>
    rasterLayer(id, {
      source: { fill: '#fff', height: 4, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 4 },
      ...overrides,
    } as Partial<CanvasLayerContract>);

  it.each<[string, CanvasNodeContract[], string, string]>([
    ['the bottom-most layer', [rasterLayer('a'), rasterLayer('b')], 'no-layer-below', 'mergeDownNoLayerBelow:b'],
    [
      'the bottom layer of a group',
      [groupContract('G', [rasterLayer('a'), rasterLayer('b')])],
      'bottom-of-group',
      'mergeDownBottomOfGroup:b',
    ],
    [
      'a layer above a group',
      [rasterLayer('b'), groupContract('Below', [rasterLayer('a')])],
      'group-below',
      'mergeDownGroupBelow:Below',
    ],
    ['a layer above a shape layer', [rasterLayer('b'), shape('a')], 'not-mergeable', 'mergeDownNotMergeable:a'],
    [
      'paint above a disabled shape layer, for its kind',
      [rasterLayer('b'), shape('a', { isEnabled: false })],
      'not-mergeable',
      'mergeDownNotMergeable:a',
    ],
    [
      'a layer above a hidden one',
      [rasterLayer('b'), rasterLayer('a', { isEnabled: false })],
      'disabled',
      'mergeDownHidden:a',
    ],
    [
      'a layer in a hidden group, naming the group',
      [groupContract('Hidden', [rasterLayer('b'), rasterLayer('a')], { isEnabled: false })],
      'disabled',
      'mergeDownHidden:Hidden',
    ],
    [
      'a layer above a locked one',
      [rasterLayer('b'), rasterLayer('a', { isLocked: true })],
      'locked',
      'mergeDownLocked:a',
    ],
    [
      'a mask, pointing to Merge masks',
      [rasterLayer('b', { type: 'inpaint_mask' } as Partial<CanvasLayerContract>)],
      'wrong-type',
      'mergeDownMaskSelected',
    ],
    ['a group', [groupContract('b', [rasterLayer('a')]), rasterLayer('c')], 'wrong-type', 'mergeDownGroupSelected'],
  ])('explains why %s cannot merge down, once per layer and reason', (_, nodes, reason, message) => {
    const document = { ...documentOf([], 'b'), stacks: stacksFrom(nodes) };
    expect(explainMergeDown(document)).toEqual([
      [`merge-down:b:${reason}`, title, `widgets.layers.actions.${message}`],
    ]);
  });
});

describe('toggleNonRasterLayers', () => {
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
});

describe('group selection', () => {
  const nodeDocument = (nodes: CanvasNodeContract[], selectedLayerId: string | null): CanvasDocumentContractV3 => ({
    ...documentOf([], selectedLayerId),
    stacks: stacksFrom(nodes),
  });

  it('reorders a selected group', () => {
    const document = nodeDocument([groupContract('g', [rasterLayer('a')]), rasterLayer('b')], 'g');
    const reorder = run('canvas.layerToBack', { document, selectedLayerIds: ['g'] });
    expect(reorder.layers.commitPrepared).toHaveBeenCalledWith(
      'widgets.canvas.commands.reorderLayer',
      expect.objectContaining({ forward: expect.objectContaining({ type: 'reorderCanvasSiblings' }) })
    );
  });
});

describe('registration tables', () => {
  const allEntries = [...CANVAS_COMPANION_HOTKEYS, ...CANVAS_ONLY_HOTKEYS, ...CANVAS_HEADER_HOTKEYS];

  it('mirrors every registered key in the hotkey catalog with the same defaults, under Canvas', () => {
    const catalogDefaults = Object.fromEntries(
      firstPartyHotkeyCatalog
        .filter((hotkey) => hotkey.category === 'canvas')
        .map((hotkey) => [hotkey.id, hotkey.defaultKeys])
    );
    const registeredDefaults = Object.fromEntries(allEntries.map(([id, , keys]) => [id, [...keys]]));

    expect(registeredDefaults).toEqual(catalogDefaults);
  });

  it('never gives two entries the same id or default chord', () => {
    const ids = allEntries.map(([id]) => id);
    const chords = allEntries.flatMap(([, , keys]) => keys);
    expect(new Set(ids).size).toBe(ids.length);
    expect(chords.filter((chord, index) => chords.indexOf(chord) !== index)).toEqual([]);
    expect(chords).toEqual(expect.arrayContaining(['mod+1', 'mod+2', 'mod+3', 'mod+4', 'mod+z', 'shift+t']));
  });
});

import type { CanvasDocumentContractV3, Rect } from '@workbench/canvas-engine/api';

import { getDocumentLeaves } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it, vi, type Mock } from 'vitest';

import {
  applyFitBbox,
  confirmNewCanvas,
  executeCanvasHeaderCommand,
  zoomAtViewportCentre,
  type CanvasHeaderCommandContext,
} from './canvasHeaderCommands';

const BBOX: Rect = { height: 64, width: 64, x: 0, y: 0 };
const FIT_LAYERS: Rect = { height: 128, width: 256, x: 8, y: 16 };
const FIT_MASKS: Rect = { height: 32, width: 32, x: 4, y: 4 };

const documentOf = (): CanvasDocumentContractV3 =>
  ({
    background: { color: '#000000', type: 'transparent' },
    bbox: BBOX,
    height: 512,
    stacks: stacksFrom([]),
    selectedLayerId: null,
    version: 3,
    width: 768,
  }) as unknown as CanvasDocumentContractV3;

const createEngine = (viewportSize = { height: 600, width: 800 }) => {
  const zoomAtPoint = vi.fn();
  const engine = {
    document: { replaceDocument: vi.fn(() => true) },
    layers: { commitStructural: vi.fn(() => ({ status: 'committed' as const })) },
    viewport: {
      fitToView: vi.fn(),
      getViewport: vi.fn(() => ({ getViewportSize: () => viewportSize, zoomAtPoint })),
    },
  };
  return { engine, zoomAtPoint };
};

const contextOf = (overrides: Partial<CanvasHeaderCommandContext> = {}): CanvasHeaderCommandContext => ({
  reportStructuralCommit: () => undefined,
  document: documentOf(),
  editingLocked: false,
  engine: createEngine().engine,
  fitLayersRect: FIT_LAYERS,
  fitMasksRect: FIT_MASKS,
  openNewCanvas: vi.fn(),
  saveToGallery: vi.fn(),
  t: (key) => key,
  ...overrides,
});

describe('zoomAtViewportCentre', () => {
  it('anchors the zoom at the viewport centre', () => {
    const { engine, zoomAtPoint } = createEngine({ height: 600, width: 800 });
    zoomAtViewportCentre(engine, 2.5);
    expect(zoomAtPoint).toHaveBeenCalledWith(2.5, { x: 400, y: 300 });
  });

  it('re-reads the viewport size on every call', () => {
    const { engine, zoomAtPoint } = createEngine({ height: 200, width: 100 });
    zoomAtViewportCentre(engine, 1);
    expect(zoomAtPoint).toHaveBeenCalledWith(1, { x: 50, y: 100 });
  });
});

describe('applyFitBbox', () => {
  it('commits one undoable setCanvasBbox whose inverse restores the current bbox', () => {
    const { engine } = createEngine();
    applyFitBbox(contextOf({ engine }), FIT_LAYERS, false);
    expect(engine.layers.commitStructural).toHaveBeenCalledWith(
      'widgets.canvas.commands.fitBbox',
      { bbox: FIT_LAYERS, type: 'setCanvasBbox' },
      { bbox: BBOX, type: 'setCanvasBbox' }
    );
  });

  it('re-fits the view only when asked', () => {
    const withRefit = createEngine();
    applyFitBbox(contextOf({ engine: withRefit.engine }), FIT_LAYERS, true);
    expect(withRefit.engine.viewport.fitToView).toHaveBeenCalled();

    const withoutRefit = createEngine();
    applyFitBbox(contextOf({ engine: withoutRefit.engine }), FIT_LAYERS, false);
    expect(withoutRefit.engine.viewport.fitToView).not.toHaveBeenCalled();
  });

  it('is inert with a null rect (nothing to fit to)', () => {
    const { engine } = createEngine();
    applyFitBbox(contextOf({ engine }), null, true);
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
    expect(engine.viewport.fitToView).not.toHaveBeenCalled();
  });

  it('is inert while editing is locked, even with a valid rect', () => {
    const { engine } = createEngine();
    applyFitBbox(contextOf({ editingLocked: true, engine }), FIT_LAYERS, true);
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
    expect(engine.viewport.fitToView).not.toHaveBeenCalled();
  });
});

describe('confirmNewCanvas', () => {
  const replaced = (engine: ReturnType<typeof createEngine>['engine']) =>
    (engine.document.replaceDocument as Mock).mock.calls[0]?.[0] as CanvasDocumentContractV3;

  it('replaces the document at the current dimensions through the engine', () => {
    const { engine } = createEngine();
    confirmNewCanvas({ document: { height: 512, width: 768 }, editingLocked: false, engine });
    expect(engine.document.replaceDocument).toHaveBeenCalledTimes(1);
    expect(replaced(engine).width).toBe(768);
    expect(replaced(engine).height).toBe(512);
  });

  it('seeds exactly one empty inpaint mask', () => {
    const { engine } = createEngine();
    confirmNewCanvas({ document: { height: 64, width: 64 }, editingLocked: false, engine });
    expect(getDocumentLeaves(replaced(engine))).toHaveLength(1);
    expect(getDocumentLeaves(replaced(engine))[0]?.type).toBe('inpaint_mask');
  });

  it('refuses while editing is locked — the second gate behind the confirm dialog', () => {
    const { engine } = createEngine();
    confirmNewCanvas({ document: { height: 512, width: 768 }, editingLocked: true, engine });
    expect(engine.document.replaceDocument).not.toHaveBeenCalled();
  });
});

describe('executeCanvasHeaderCommand', () => {
  it('fitBboxToLayers commits the layers rect and re-fits', () => {
    const { engine } = createEngine();
    executeCanvasHeaderCommand('canvas.fitBboxToLayers', contextOf({ engine }));
    expect(engine.layers.commitStructural).toHaveBeenCalledWith(
      'widgets.canvas.commands.fitBbox',
      { bbox: FIT_LAYERS, type: 'setCanvasBbox' },
      { bbox: BBOX, type: 'setCanvasBbox' }
    );
    expect(engine.viewport.fitToView).toHaveBeenCalled();
  });

  it('fitBboxToMasks commits the masks rect and does NOT re-fit', () => {
    const { engine } = createEngine();
    executeCanvasHeaderCommand('canvas.fitBboxToMasks', contextOf({ engine }));
    expect(engine.layers.commitStructural).toHaveBeenCalledWith(
      'widgets.canvas.commands.fitBbox',
      { bbox: FIT_MASKS, type: 'setCanvasBbox' },
      { bbox: BBOX, type: 'setCanvasBbox' }
    );
    expect(engine.viewport.fitToView).not.toHaveBeenCalled();
  });

  it('newSession opens the confirm dialog instead of replacing directly', () => {
    const { engine } = createEngine();
    const openNewCanvas = vi.fn();
    executeCanvasHeaderCommand('canvas.newSession', contextOf({ engine, openNewCanvas }));
    expect(openNewCanvas).toHaveBeenCalled();
    expect(engine.document.replaceDocument).not.toHaveBeenCalled();
  });

  it('fit commands are inert with no rect to fit to', () => {
    const { engine } = createEngine();
    const ctx = contextOf({ engine, fitLayersRect: null, fitMasksRect: null });
    executeCanvasHeaderCommand('canvas.fitBboxToLayers', ctx);
    executeCanvasHeaderCommand('canvas.fitBboxToMasks', ctx);
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
  });

  it.each([
    ['canvas.saveToGallery', 'canvas'],
    ['canvas.saveBboxToGallery', 'bbox'],
  ] as const)('%s saves the %s region, unless an operation owns the document', (commandId, region) => {
    const saveToGallery = vi.fn();
    executeCanvasHeaderCommand(commandId, contextOf({ editingLocked: true, saveToGallery }));
    expect(saveToGallery).not.toHaveBeenCalled();
    executeCanvasHeaderCommand(commandId, contextOf({ saveToGallery }));
    expect(saveToGallery).toHaveBeenCalledWith(region);
  });

  it('ignores an unknown command id', () => {
    const openNewCanvas = vi.fn();
    const { engine } = createEngine();
    executeCanvasHeaderCommand('canvas.nonexistent', contextOf({ engine, openNewCanvas }));
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
    expect(openNewCanvas).not.toHaveBeenCalled();
    expect(engine.document.replaceDocument).not.toHaveBeenCalled();
  });
});

import type { CanvasDocumentContractV2, Rect } from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

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

const documentOf = (): CanvasDocumentContractV2 =>
  ({
    background: { color: '#000000', type: 'transparent' },
    bbox: BBOX,
    height: 512,
    layers: [],
    selectedLayerId: null,
    version: 2,
    width: 768,
  }) as unknown as CanvasDocumentContractV2;

const createEngine = (viewportSize = { height: 600, width: 800 }) => {
  const zoomAtPoint = vi.fn();
  const engine = {
    layers: { commitStructural: vi.fn(() => true) },
    viewport: {
      fitToView: vi.fn(),
      getViewport: vi.fn(() => ({ getViewportSize: () => viewportSize, zoomAtPoint })),
    },
  };
  return { engine, zoomAtPoint };
};

let dispatch: Mock<(mutation: CanvasProjectMutation) => boolean>;

const contextOf = (overrides: Partial<CanvasHeaderCommandContext> = {}): CanvasHeaderCommandContext => ({
  dispatch,
  document: documentOf(),
  editingLocked: false,
  engine: createEngine().engine,
  fitLayersRect: FIT_LAYERS,
  fitMasksRect: FIT_MASKS,
  openNewCanvas: vi.fn(),
  t: (key) => key,
  ...overrides,
});

beforeEach(() => {
  dispatch = vi.fn(() => true);
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
  it('replaces the document at the current dimensions', () => {
    confirmNewCanvas({ dispatch, document: { height: 512, width: 768 }, editingLocked: false });
    expect(dispatch).toHaveBeenCalledTimes(1);
    const mutation = dispatch.mock.calls[0]?.[0] as Extract<CanvasProjectMutation, { type: 'replaceCanvasDocument' }>;
    expect(mutation.type).toBe('replaceCanvasDocument');
    expect(mutation.document.width).toBe(768);
    expect(mutation.document.height).toBe(512);
  });

  it('seeds exactly one empty inpaint mask', () => {
    confirmNewCanvas({ dispatch, document: { height: 64, width: 64 }, editingLocked: false });
    const mutation = dispatch.mock.calls[0]?.[0] as Extract<CanvasProjectMutation, { type: 'replaceCanvasDocument' }>;
    expect(mutation.document.layers).toHaveLength(1);
    expect(mutation.document.layers[0]?.type).toBe('inpaint_mask');
  });

  it('refuses while editing is locked — the second gate behind the confirm dialog', () => {
    confirmNewCanvas({ dispatch, document: { height: 512, width: 768 }, editingLocked: true });
    expect(dispatch).not.toHaveBeenCalled();
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
    const openNewCanvas = vi.fn();
    executeCanvasHeaderCommand('canvas.newSession', contextOf({ openNewCanvas }));
    expect(openNewCanvas).toHaveBeenCalled();
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('fit commands are inert with no rect to fit to', () => {
    const { engine } = createEngine();
    const ctx = contextOf({ engine, fitLayersRect: null, fitMasksRect: null });
    executeCanvasHeaderCommand('canvas.fitBboxToLayers', ctx);
    executeCanvasHeaderCommand('canvas.fitBboxToMasks', ctx);
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
  });

  it('ignores an unknown command id', () => {
    const openNewCanvas = vi.fn();
    const { engine } = createEngine();
    executeCanvasHeaderCommand('canvas.nonexistent', contextOf({ engine, openNewCanvas }));
    expect(engine.layers.commitStructural).not.toHaveBeenCalled();
    expect(openNewCanvas).not.toHaveBeenCalled();
    expect(dispatch).not.toHaveBeenCalled();
  });
});

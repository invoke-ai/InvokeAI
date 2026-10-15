import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { StrokeCommittedEvent } from '@workbench/canvas-engine/tools/tool';

import { createCanvasMutationContext } from '@workbench/canvas-engine/controllers/mutationContext';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createHistory, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import { createStrokeEdits } from './strokeCommit';

const RECT = { height: 4, width: 4, x: 1, y: 2 };
const pixels = (): ImageData => ({ data: new Uint8ClampedArray(64), height: 4, width: 4 }) as unknown as ImageData;

const strokeEvent = (overrides: Partial<StrokeCommittedEvent> = {}): StrokeCommittedEvent => ({
  afterImageData: pixels(),
  beforeImageData: pixels(),
  dirtyRect: RECT,
  layerId: 'paint',
  tool: 'brush',
  ...overrides,
});

/** The stroke edits over a real reducer, mutation context and history. */
const createHarness = (byteBudget?: number, gesture = { active: false }) => {
  let project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
    document: documentFrom([layerContract('paint')]),
    type: 'replaceCanvasDocument',
  });
  const history = createHistory({ byteBudget });
  const layerCache = createLayerCacheStore(createTestStubRasterBackend());
  const context = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'unused',
    dispatch: (action: CanvasProjectMutation) => {
      project = applyCanvasProjectMutation(project, action);
      return true;
    },
    editOwner: Symbol('owner'),
    editingLocked: { get: () => false, subscribe: () => () => undefined },
    getDocument: () => project.canvas.document,
    getReducerDocument: () => project.canvas.document,
    history,
    installPrepared: () => undefined,
    isGestureActive: () => gesture.active,
    isGuardCurrent: () => true,
    preparePixels: () => {
      throw new Error('unused');
    },
    projectId: project.id,
    refreshMirror: () => undefined,
    reserveRaster: () => ({ lease: { release: () => undefined }, status: 'ok' }),
    subscribeReducer: () => () => undefined,
  });
  const deps = {
    applyImagePatch: vi.fn(() => Promise.resolve()),
    commitPaintEdit: vi.fn(),
    ctx: context,
    layerCache,
    markLayerDirty: vi.fn(),
    notifyLayerPainted: vi.fn(),
    reportRefusal: vi.fn(),
    strokeListeners: new Set<(event: StrokeCommittedEvent) => void>(),
  };
  return {
    deps,
    document: () => project.canvas.document,
    dispatch: (action: CanvasProjectMutation) => {
      project = applyCanvasProjectMutation(project, action);
    },
    edits: createStrokeEdits(deps),
    history,
  };
};

describe('stroke edits', () => {
  it('records the painted pixels as one labelled step and publishes the paint afterwards', () => {
    const h = createHarness();
    const listener = vi.fn();
    h.deps.strokeListeners.add(listener);
    const event = strokeEvent({ tool: 'eraser' });

    const edit = h.edits.begin()!;
    expect(edit.grow(128)).toBe(true);
    expect(edit.commit(event)).toBe(true);

    expect(h.history.entries().past).toEqual(['Eraser stroke']);
    expect(h.history.byteSize()).toBe(128);
    expect(h.deps.notifyLayerPainted).toHaveBeenCalledWith('paint');
    expect(h.deps.markLayerDirty).toHaveBeenCalledWith('paint');
    expect(h.deps.commitPaintEdit).toHaveBeenCalledOnce();
    expect(listener).toHaveBeenCalledWith(event);
  });

  it('admits a stroke during its own gesture and records it once the gesture ends', () => {
    const gesture = { active: true };
    const h = createHarness(undefined, gesture);

    const edit = h.edits.begin()!;
    expect(edit).not.toBeNull();
    edit.grow(128);
    gesture.active = false;
    expect(edit.commit(strokeEvent())).toBe(true);
    expect(h.history.canUndo()).toBe(true);
  });

  it('refuses a stroke whose undo footprint could never be kept, before it paints', () => {
    const h = createHarness(HISTORY_ENTRY_OVERHEAD_BYTES + 64);

    expect(h.edits.begin(HISTORY_ENTRY_OVERHEAD_BYTES + 65)).toBeNull();
    expect(h.deps.reportRefusal).toHaveBeenCalledWith('over-budget');

    const edit = h.edits.begin()!;
    expect(edit.grow(64)).toBe(true);
    expect(edit.grow(1)).toBe(false);
    expect(edit.grow(1)).toBe(false);
    expect(h.deps.reportRefusal).toHaveBeenCalledTimes(2);
    edit.cancel();
    expect(h.history.canUndo()).toBe(false);
  });

  it('replays the before and after pixels through the engine bridge', async () => {
    const h = createHarness();
    const event = strokeEvent();
    const edit = h.edits.begin()!;
    edit.grow(128);
    edit.commit(event);

    await h.history.undo();
    expect(h.deps.applyImagePatch).toHaveBeenLastCalledWith('paint', RECT, event.beforeImageData);
    await h.history.redo();
    expect(h.deps.applyImagePatch).toHaveBeenLastCalledWith('paint', RECT, event.afterImageData);
  });

  it('keeps a step in place when its pixels cannot be restored', async () => {
    const h = createHarness();
    const edit = h.edits.begin()!;
    edit.grow(128);
    edit.commit(strokeEvent());
    h.deps.applyImagePatch.mockRejectedValueOnce(new Error('not ready'));

    expect((await h.history.undo()).status).toBe('failed');
    expect(h.history.entries()).toEqual({ future: [], past: ['Brush stroke'] });
  });
});

describe('a stroke that auto-created its layer', () => {
  const createdStroke = (h: ReturnType<typeof createHarness>) => {
    const anchor = stackTopAnchor(h.deps.ctx.projectId);
    h.dispatch({ anchor, layer: layerContract('created') as CanvasLayerContract, type: 'addCanvasLayer' });
    const created = getDocumentLayer(h.document(), 'created')!;
    const edit = h.edits.begin()!;
    edit.grow(128);
    edit.commit(strokeEvent({ createdLayer: { anchor, layer: created }, layerId: 'created' }));
    return created;
  };

  it('undoes by removing the layer and redoes by restoring it with its pixels', async () => {
    const h = createHarness();
    const created = createdStroke(h);

    await h.history.undo();
    expect(getDocumentLayer(h.document(), 'created')).toBeNull();

    await h.history.redo();
    expect(getDocumentLayer(h.document(), 'created')).toEqual(created);
    expect(h.deps.layerCache.peek('created')?.stale).toBe(false);
    expect(h.deps.applyImagePatch).toHaveBeenLastCalledWith('created', RECT, expect.anything());
  });

  it('takes the layer back out when a redo cannot restore its pixels, so the redo stays retryable', async () => {
    const h = createHarness();
    const created = createdStroke(h);
    await h.history.undo();
    h.deps.applyImagePatch.mockRejectedValueOnce(new Error('not ready'));

    expect((await h.history.redo()).status).toBe('failed');
    expect(getDocumentLayer(h.document(), 'created')).toBeNull();
    expect(h.history.canRedo()).toBe(true);
    expect((await h.history.redo()).status).toBe('applied');
    expect(getDocumentLayer(h.document(), 'created')).toEqual(created);
  });

  it('accounts for both pixel buffers plus overhead', () => {
    const h = createHarness();
    createdStroke(h);
    expect(h.history.byteSize()).toBe(64 + 64 + HISTORY_ENTRY_OVERHEAD_BYTES);
  });
});

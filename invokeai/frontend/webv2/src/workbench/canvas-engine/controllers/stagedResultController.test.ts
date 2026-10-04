import type { CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { Project } from '@workbench/projectContracts';

import { getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createHistory, NO_HELD_ASSET_REFS } from '@workbench/canvas-engine/history/history';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import { createCanvasMutationContext } from './mutationContext';
import { StagedResultController } from './stagedResultController';

const candidate: CanvasStagingCandidateContract = {
  height: 40,
  imageName: 'result.png',
  imageUrl: '/result.png',
  placement: { height: 80, opacity: 0.5, width: 60, x: 12, y: 18 },
  queuedAt: '2026-07-16T00:00:00.000Z',
  sourceQueueItemId: 'queue-1',
  thumbnailUrl: '/thumb.png',
  width: 30,
};
const selection = { candidate, selectedImageIndex: 0 } as const;

interface HarnessOptions {
  readonly byteBudget?: number;
  readonly gestureActive?: boolean;
  readonly locked?: boolean;
  /** The reducer keeps the project unchanged, as when the candidate moved on. */
  readonly rejectDispatch?: boolean;
  /** Mirror refresh fails, so an accepted mutation can never be mirrored. */
  readonly mirrorBroken?: boolean;
  readonly staged?: boolean;
}

/** A real reducer, history and transaction protocol around the controller; the mirror follows the reducer. */
const createHarness = (options: HarnessOptions = {}) => {
  const base = createInitialWorkbenchState().projects[0]!;
  let project: Project = {
    ...base,
    canvas: {
      ...base.canvas,
      stagingArea: {
        ...base.canvas.stagingArea,
        isVisible: true,
        pendingImageIds: options.staged === false ? [] : [candidate.imageName],
        pendingImages: options.staged === false ? [] : [candidate],
        selectedImageIndex: 0,
      },
    },
  };
  let mirrorBroken = options.mirrorBroken ?? false;
  let mirrorDocument = project.canvas.document;
  const dispatched: CanvasProjectMutation[] = [];
  const listeners = new Set<() => void>();
  const history = createHistory({ byteBudget: options.byteBudget });
  const dispatch = (mutation: CanvasProjectMutation): boolean => {
    dispatched.push(mutation);
    if (options.rejectDispatch) {
      return false;
    }
    const next = applyCanvasProjectMutation(project, mutation);
    const changed = next !== project;
    project = next;
    if (!mirrorBroken) {
      mirrorDocument = project.canvas.document;
    }
    listeners.forEach((listener) => listener());
    return changed;
  };
  const ctx = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'layer-1',
    dispatch,
    editOwner: Symbol('owner'),
    editingLocked: { get: () => options.locked ?? false, subscribe: () => () => undefined },
    getDocument: () => mirrorDocument,
    getReducerDocument: () => project.canvas.document,
    history,
    installPrepared: () => undefined,
    isGestureActive: () => options.gestureActive ?? false,
    isGuardCurrent: () => true,
    preparePixels: () => ({}) as never,
    projectId: base.id,
    refreshMirror: () => {
      if (mirrorBroken) {
        throw new Error('mirror broken');
      }
      mirrorDocument = project.canvas.document;
    },
    reserveRaster: () => ({ lease: { release: () => undefined }, status: 'ok' }),
    subscribeReducer: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  });
  const controller = new StagedResultController({
    createEventId: () => 'event-1',
    ctx,
    getCanvasState: () => project.canvas,
    now: () => '2026-07-16T01:00:00.000Z',
  });
  return {
    breakMirror: (broken: boolean) => {
      mirrorBroken = broken;
    },
    controller,
    dispatched,
    history,
    project: () => project,
  };
};

describe('StagedResultController', () => {
  it('commits the selected candidate as one undo step that replays both ways', async () => {
    const h = createHarness();
    const initial = h.project().canvas.document;

    expect(h.controller.commit(selection)).toEqual({ layerId: 'layer-1', status: 'committed' });
    expect(getDocumentLayer(h.project().canvas.document, 'layer-1')).toMatchObject({
      id: 'layer-1',
      opacity: 0.5,
      source: { image: { imageName: 'result.png' }, type: 'image' },
      transform: { scaleX: 2, scaleY: 2, x: 12, y: 18 },
      type: 'raster',
    });
    expect(h.project().canvas.stagingArea.pendingImages).toEqual([]);

    expect(await h.history.undo()).toEqual({ status: 'applied' });
    expect(getDocumentLeaves(h.project().canvas.document)).toEqual(getDocumentLeaves(initial));
    expect(h.project().canvas.document.selectedLayerId).toBe(initial.selectedLayerId);
    expect(h.project().canvas.stagingArea.pendingImages).toEqual([]);

    expect(await h.history.redo()).toEqual({ status: 'applied' });
    expect(getDocumentLayer(h.project().canvas.document, 'layer-1')).not.toBeNull();
    expect(h.project().canvas.document.selectedLayerId).toBe('layer-1');
  });

  it('returns stale without history when the reducer leaves the project unchanged', () => {
    const h = createHarness({ rejectDispatch: true });

    expect(h.controller.commit(selection)).toEqual({ status: 'stale' });
    expect(h.project().canvas.stagingArea.pendingImages).toEqual([candidate]);
    expect(h.history.canUndo()).toBe(false);
  });

  it('rolls the accepted commit back when the mirror cannot follow it', () => {
    const h = createHarness({ mirrorBroken: true });
    const before = h.project().canvas;

    expect(h.controller.commit(selection)).toEqual({ status: 'stale' });
    expect(getDocumentLeaves(h.project().canvas.document)).toEqual(getDocumentLeaves(before.document));
    expect(h.project().canvas.stagingArea).toBe(before.stagingArea);
    expect(h.dispatched.map((mutation) => mutation.type)).toEqual(['commitStagedImage', 'rollbackStagedImageCommit']);
    expect(h.history.canUndo()).toBe(false);
  });

  it.each([
    { name: 'editing is locked', options: { locked: true } },
    { name: 'a gesture is active', options: { gestureActive: true } },
  ])('returns busy without mutation when $name', ({ options }) => {
    const h = createHarness(options);

    expect(h.controller.commit(selection)).toEqual({ status: 'busy' });
    expect(h.dispatched).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
  });

  it('refuses before mutating when history could never retain the step', () => {
    const h = createHarness({ byteBudget: 16 });

    expect(h.controller.commit(selection)).toEqual({ status: 'over-budget' });
    expect(h.dispatched).toEqual([]);
    expect(h.project().canvas.stagingArea.pendingImages).toEqual([candidate]);
  });

  it('keeps a failed undo on the stack with the document unchanged', async () => {
    const h = createHarness();
    h.controller.commit(selection);
    const committed = getDocumentLeaves(h.project().canvas.document);
    h.breakMirror(true);

    expect((await h.history.undo()).status).toBe('failed');
    expect(h.history.canUndo()).toBe(true);
    expect(h.history.canRedo()).toBe(false);
    expect(getDocumentLeaves(h.project().canvas.document)).toEqual(committed);
  });

  it('clears the redo stack after a new staged image commit', async () => {
    const h = createHarness();
    h.history
      .admit(1)!
      .publish({ bytes: 1, heldAssetRefs: NO_HELD_ASSET_REFS, label: 'older edit', redo: vi.fn(), undo: vi.fn() });
    await h.history.undo();
    expect(h.history.canRedo()).toBe(true);

    expect(h.controller.commit(selection).status).toBe('committed');
    expect(h.history.canRedo()).toBe(false);
  });

  it.each([
    ['edits are locked', { locked: true }],
    ['a gesture is active', { gestureActive: true }],
  ])('refuses without mutating staging or history while %s', (_, options) => {
    const h = createHarness(options);
    const before = h.project();

    expect(h.controller.commit(selection).status).not.toBe('committed');
    expect(h.dispatched).toEqual([]);
    expect(h.project()).toBe(before);
    expect(h.history.canUndo()).toBe(false);
  });

  it('returns missing without mutation when the candidate is no longer staged', () => {
    const h = createHarness({ staged: false });

    expect(h.controller.commit(selection)).toEqual({ status: 'missing' });
    expect(h.dispatched).toEqual([]);
  });

  it('rejects retained commit capabilities after disposal', () => {
    const h = createHarness();
    h.controller.dispose();

    expect(h.controller.commit(selection)).toEqual({ status: 'missing' });
    expect(h.dispatched).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
  });
});

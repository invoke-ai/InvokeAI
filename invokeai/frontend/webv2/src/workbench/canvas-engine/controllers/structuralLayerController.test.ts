import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';

import { createDocumentModel, type CanvasDocumentModel } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentIndex, getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { stackTopAnchor } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { haveSameStructure } from '@workbench/canvas-engine/document/layerStacks';
import { createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { describe, expect, it } from 'vitest';

import { createStructuralEngineStub } from './structuralEngine.testStub';

const createHarness = createStructuralEngineStub;

const rename = (id: string, name: string): Extract<CanvasProjectMutation, { type: 'updateCanvasLayer' }> => ({
  id,
  patch: { name },
  type: 'updateCanvasLayer',
});
const layerName = (document: CanvasDocumentContractV3, id = 'layer'): string | undefined =>
  getDocumentLeaves(document).find((layer) => layer.id === id)?.name;

describe('StructuralLayerController', () => {
  it('coalesces previews to one dispatch per flush, last value wins', () => {
    let flush: (() => void) | null = null;
    const harness = createHarness({
      schedulePreview: (callback) => {
        flush = callback;
        return () => undefined;
      },
    });
    const session = harness.controller.beginPreview()!;
    expect(session.apply(rename('layer', 'A'))).toBe(true);
    expect(session.apply(rename('layer', 'B'))).toBe(true);
    expect(harness.dispatched).toHaveLength(0);
    flush!();
    expect(harness.dispatched).toHaveLength(1);
    expect(layerName(harness.document())).toBe('B');
  });

  it('discards a pending preview when a commit lands, and on dispose', () => {
    let cancelled = 0;
    const harness = createHarness({
      schedulePreview: () => () => {
        cancelled += 1;
      },
    });
    harness.controller.beginPreview()!.apply(rename('layer', 'Stale'));
    expect(harness.controller.commit('Rename', rename('layer', 'Final'), rename('layer', 'Layer'))).toMatchObject({
      status: 'committed',
    });
    expect(cancelled).toBe(1);
    expect(layerName(harness.document())).toBe('Final');
    harness.controller.beginPreview()!.apply(rename('layer', 'Orphan'));
    harness.controller.dispose();
    expect(cancelled).toBe(2);
    expect(layerName(harness.document())).toBe('Final');
  });

  it('dispatches previews synchronously without an animation frame', () => {
    const harness = createHarness();
    expect(harness.controller.beginPreview()!.apply(rename('layer', 'Live'))).toBe(true);
    expect(layerName(harness.document())).toBe('Live');
  });

  it('commits through the guarded dispatch and records one failure-atomic history entry', async () => {
    const { controller, document, history, mirror } = createHarness();

    expect(controller.canCommit()).toBe(true);
    expect(controller.commit('Rename', rename('layer', 'Renamed'), rename('layer', 'Layer'))).toEqual({
      status: 'committed',
    });
    expect(layerName(document())).toBe('Renamed');
    expect(mirror()).toBe(document());
    expect(history.canUndo()).toBe(true);

    await history.undo();
    expect(layerName(document())).toBe('Layer');
    await history.redo();
    expect(layerName(document())).toBe('Renamed');
  });

  it.each([
    { expected: 'busy', locked: true, reason: 'editing is locked' },
    { expected: 'gesture-active', gestureActive: true, reason: 'a gesture is active' },
  ])('refuses before dispatch when $reason', ({ expected, ...options }) => {
    const { controller, dispatched, history } = createHarness(options);

    expect(controller.commit('Rename', rename('layer', 'Renamed'), rename('layer', 'Layer'))).toEqual({
      status: expected,
    });
    expect(dispatched).toEqual([]);
    expect(history.canUndo()).toBe(false);
  });

  it('refuses after disposal', () => {
    const { controller, dispatched } = createHarness();

    controller.dispose();

    expect(controller.canCommit()).toBe(false);
    expect(controller.commit('Rename', rename('layer', 'Renamed'), rename('layer', 'Layer'))).toEqual({
      status: 'not-ready',
    });
    expect(dispatched).toEqual([]);
  });

  it('refuses an edit prepared against an older revision as stale', () => {
    const { controller, ctx, dispatched } = createHarness();
    const captured = ctx.getEditRevision();

    expect(controller.commit('First', rename('layer', 'A'), rename('layer', 'Layer'))).toEqual({ status: 'committed' });
    const actual = ctx.getEditRevision();

    expect(
      controller.commit('Second', rename('layer', 'B'), rename('layer', 'A'), { expectedRevision: captured })
    ).toEqual({ actualRevision: actual, expectedRevision: captured, status: 'stale' });
    expect(dispatched).toHaveLength(1);
    expect(
      controller.commit('Third', rename('layer', 'B'), rename('layer', 'A'), { expectedRevision: actual })
    ).toEqual({
      status: 'committed',
    });
  });

  it('reports a reducer rejection without recording history', () => {
    const { controller, document, history } = createHarness();
    const before = document();

    expect(
      controller.commit('Select', { id: 'missing', type: 'setCanvasSelectedLayer' }, rename('layer', 'Layer'))
    ).toEqual({ status: 'dispatch-rejected' });
    expect(document()).toBe(before);
    expect(history.canUndo()).toBe(false);
  });

  it('applies the inverse when an accepted edit fails its postcondition', () => {
    const { controller, document, history, mirror } = createHarness();

    expect(
      controller.commit('Rename', rename('layer', 'Renamed'), rename('layer', 'Layer'), { verify: () => false })
    ).toEqual({ recovered: 'reverted', status: 'postcondition-failed' });
    expect(layerName(document())).toBe('Layer');
    expect(mirror()).toBe(document());
    expect(history.canUndo()).toBe(false);
  });

  it('reports when the inverse cannot be applied and leaves the edit unrecorded', () => {
    const { controller, document, history, report } = createHarness();

    expect(
      controller.commit('Rename', rename('layer', 'Renamed'), rename('missing', 'Layer'), { verify: () => false })
    ).toEqual({ recovered: 'unreverted', status: 'postcondition-failed' });
    expect(layerName(document())).toBe('Renamed');
    expect(history.canUndo()).toBe(false);
    expect(report).toHaveBeenCalledWith('Structural edit could not be reverted', 'Rename', expect.any(Error));
  });

  it('reconciles a lagging mirror before recording the entry', () => {
    const { controller, document, mirror } = createHarness({ mirrorLag: true });

    expect(controller.commit('Rename', rename('layer', 'Renamed'), rename('layer', 'Layer'))).toEqual({
      status: 'committed',
    });
    expect(mirror()).toBe(document());
  });

  it('reverts an accepted edit that can never be mirrored and says the view may lag', () => {
    const { controller, document, history, report } = createHarness({ mirrorBroken: true });

    expect(controller.commit('Rename', rename('layer', 'Renamed'), rename('layer', 'Layer'))).toEqual({
      recovered: 'reverted-unmirrored',
      status: 'postcondition-failed',
    });
    expect(layerName(document())).toBe('Layer');
    expect(history.canUndo()).toBe(false);
    expect(report).toHaveBeenCalledWith('Structural edit could not be mirrored', 'Rename', expect.any(Error));
  });

  describe('commitPrepared', () => {
    const prepareRename = (model: CanvasDocumentModel, name: string) => {
      const result = model.prepare({ id: 'layer', patch: { name }, type: 'patch' });
      if (result.status !== 'prepared') {
        throw new Error(`expected a prepared edit, got ${result.status}`);
      }
      return result.edit;
    };

    it('applies a prepared edit, verifies its postconditions and records one entry', async () => {
      const { controller, ctx, document, history, projectId } = createHarness();
      const model = createDocumentModel(document(), { editRevision: ctx.getEditRevision(), projectId });

      expect(controller.commitPrepared('Rename', prepareRename(model, 'Renamed'))).toEqual({ status: 'committed' });
      expect(layerName(document())).toBe('Renamed');
      expect(history.canUndo()).toBe(true);
      await history.undo();
      expect(layerName(document())).toBe('Layer');
    });

    it('refuses an edit prepared against an older revision or another project', () => {
      const { controller, ctx, document, history, projectId } = createHarness();
      const stale = prepareRename(
        createDocumentModel(document(), { editRevision: ctx.getEditRevision(), projectId }),
        'A'
      );
      ctx.dispatch(rename('layer', 'Elsewhere'), 'system');

      expect(controller.commitPrepared('Rename', stale)).toMatchObject({ status: 'stale' });
      const foreign = prepareRename(
        createDocumentModel(document(), { editRevision: ctx.getEditRevision(), projectId: 'other' }),
        'B'
      );
      expect(controller.commitPrepared('Rename', foreign)).toEqual({ status: 'dispatch-rejected' });
      expect(layerName(document())).toBe('Elsewhere');
      expect(history.canUndo()).toBe(false);
    });

    it('records a previewed gesture as one step without dispatching its final value again', async () => {
      const { controller, ctx, dispatched, document, history, projectId } = createHarness();
      const edit = prepareRename(
        createDocumentModel(document(), { editRevision: ctx.getEditRevision(), projectId }),
        'Live'
      );
      const session = controller.beginPreview()!;
      session.apply(rename('layer', 'Draft'));
      session.apply(rename('layer', 'Live'));
      const previews = dispatched.length;

      expect(session.commit('Rename', edit)).toEqual({ status: 'committed' });
      expect(dispatched).toHaveLength(previews);
      expect(history.entries().past).toEqual(['Rename']);
      await history.undo();
      expect(layerName(document())).toBe('Layer');
    });

    it('restores the baseline on cancel and refuses a session a newer one replaced', () => {
      const { controller, ctx, document, history, projectId } = createHarness();
      const first = controller.beginPreview()!;
      first.apply(rename('layer', 'Draft'));
      const second = controller.beginPreview()!;

      expect(first.apply(rename('layer', 'Ignored'))).toBe(false);
      expect(
        first.commit(
          'Rename',
          prepareRename(createDocumentModel(document(), { editRevision: ctx.getEditRevision(), projectId }), 'X')
        )
      ).toEqual({ status: 'busy' });
      second.apply(rename('layer', 'Other'));
      second.cancel();
      expect(layerName(document())).toBe('Layer');
      expect(history.canUndo()).toBe(false);
    });

    it('returns to the baseline when a previewed gesture is refused, even under a lock', () => {
      const locked = { value: false };
      const { controller, ctx, document, history, projectId } = createHarness({ locked });
      const edit = prepareRename(
        createDocumentModel(document(), { editRevision: ctx.getEditRevision(), projectId }),
        'Live'
      );
      const session = controller.beginPreview()!;
      session.apply(rename('layer', 'Live'));
      locked.value = true;

      expect(session.commit('Rename', edit)).toEqual({ status: 'busy' });
      expect(layerName(document())).toBe('Layer');
      expect(history.canUndo()).toBe(false);

      locked.value = false;
      const cancelled = controller.beginPreview()!;
      cancelled.apply(rename('layer', 'Draft'));
      locked.value = true;
      cancelled.cancel();
      expect(layerName(document())).toBe('Layer');
    });

    it('skips history for an edit whose policy is none', () => {
      const { controller, ctx, document, history, projectId } = createHarness();
      const model = createDocumentModel(document(), { editRevision: ctx.getEditRevision(), projectId });
      const result = model.prepare({ id: null, type: 'select' });
      if (result.status !== 'prepared') {
        throw new Error('expected a prepared selection change');
      }

      expect(controller.commitPrepared('Select', result.edit)).toEqual({ status: 'committed' });
      expect(document().selectedLayerId).toBeNull();
      expect(history.canUndo()).toBe(false);
    });
  });

  it('refuses an insertion anchored at an older edit revision as stale', () => {
    const { controller, ctx, document, projectId } = createHarness();
    const anchor = ctx.captureInsertionAnchor('raster', document().selectedLayerId);
    ctx.dispatch(rename('layer', 'Renamed'), 'system');
    const added = createEmptyPaintLayer('Added', 'added');

    expect(
      controller.commit(
        'Add',
        { anchor, layer: added, type: 'addCanvasLayer' },
        { ids: ['added'], type: 'removeCanvasLayers' }
      )
    ).toEqual({
      actualRevision: anchor.capturedEditRevision + 1,
      expectedRevision: anchor.capturedEditRevision,
      status: 'stale',
    });
    expect(getDocumentLeaves(document()).some((layer) => layer.id === 'added')).toBe(false);
    expect(anchor.projectId).toBe(projectId);
  });

  it('moves a replay the reducer refuses as a reported no-op instead of wedging history', async () => {
    const { controller, ctx, document, history, projectId, report } = createHarness();
    const added = createEmptyPaintLayer('Added', 'added');

    controller.commit(
      'Add',
      { anchor: stackTopAnchor(projectId), layer: added, type: 'addCanvasLayer' },
      { ids: ['added'], type: 'removeCanvasLayers' }
    );
    ctx.dispatch({ ids: ['added'], type: 'removeCanvasLayers' }, 'system');

    await expect(history.undo()).resolves.toEqual({ status: 'applied' });
    expect(report).toHaveBeenCalledWith('Structural history replay was refused', 'Add', expect.any(Error));
    expect(history.canUndo()).toBe(false);
    expect(history.canRedo()).toBe(true);
    expect(layerName(document(), 'added')).toBeUndefined();
  });

  it('coalesces rapid nudges into one entry and reports an ineligible nudge as rejected', async () => {
    let now = 0;
    const { controller, document, history } = createHarness({ now: () => now });

    expect(controller.nudge(1, 0)).toEqual({ status: 'committed' });
    now = 100;
    expect(controller.nudge(1, 0)).toEqual({ status: 'committed' });
    expect(getDocumentLayer(document(), 'layer')?.transform.x).toBe(2);

    await history.undo();
    expect(getDocumentLayer(document(), 'layer')?.transform.x).toBe(0);
    expect(history.canUndo()).toBe(false);

    controller.commit(
      'Deselect',
      { id: null, type: 'setCanvasSelectedLayer' },
      { id: 'layer', type: 'setCanvasSelectedLayer' }
    );
    expect(controller.nudge(1, 0)).toEqual({ status: 'dispatch-rejected' });
  });
});

describe('hierarchy recovery', () => {
  it('reverts a reparent whose postconditions fail and leaves the tree, selection and history untouched', async () => {
    const { controller, ctx, document, history, projectId } = createHarness();
    const inner = createEmptyPaintLayer('Inner', 'inner');
    const group = {
      children: [inner],
      id: 'g',
      isEnabled: true,
      isLocked: false,
      name: 'Group',
      type: 'group' as const,
    };
    ctx.dispatch(
      {
        add: [{ anchor: stackTopAnchor(projectId), nodes: [group] }],
        enabledUpdates: [],
        type: 'applyCanvasLayerStackMutation',
      },
      'system'
    );
    const before = document();
    const parentOf = (id: string) => getDocumentIndex(document()).byId.get(id)?.parentId;
    expect(parentOf('layer')).toBeNull();

    const model = createDocumentModel(before, { editRevision: ctx.getEditRevision(), projectId });
    const result = model.prepare({ beforeId: null, ids: ['layer'], parentId: 'g', type: 'reparent' });
    if (result.status !== 'prepared') {
      throw new Error(result.status);
    }
    // Fault injection: the edit lands but claims an order the reducer did not produce.
    const tampered = {
      ...result.edit,
      postconditions: [
        { kind: 'sibling-order' as const, orderedIds: ['layer', 'inner'], parentId: 'g', stack: 'raster' as const },
      ],
    };
    expect(controller.commitPrepared('Reparent', tampered)).toEqual({
      recovered: 'reverted',
      status: 'postcondition-failed',
    });
    expect(haveSameStructure(document().stacks, before.stacks)).toBe(true);
    expect(document().selectedLayerId).toBe(before.selectedLayerId);
    expect(history.canUndo()).toBe(false);

    // Prepared afresh against the reverted document, the same edit lands and undoes exactly.
    const fresh = createDocumentModel(document(), { editRevision: ctx.getEditRevision(), projectId }).prepare({
      beforeId: null,
      ids: ['layer'],
      parentId: 'g',
      type: 'reparent',
    });
    if (fresh.status !== 'prepared') {
      throw new Error(fresh.status);
    }
    expect(controller.commitPrepared('Reparent', fresh.edit)).toEqual({ status: 'committed' });
    expect(parentOf('layer')).toBe('g');
    await history.undo();
    expect(haveSameStructure(document().stacks, before.stacks)).toBe(true);
  });
});

describe('preview sessions ended by the engine', () => {
  const prepareRename = (harness: ReturnType<typeof createHarness>, name: string) => {
    const model = createDocumentModel(harness.document(), {
      editRevision: harness.ctx.getEditRevision(),
      projectId: harness.projectId,
    });
    const result = model.prepare({ id: 'layer', patch: { name }, type: 'patch' });
    if (result.status !== 'prepared') {
      throw new Error(`expected a prepared edit, got ${result.status}`);
    }
    return result.edit;
  };

  it('restores a previewed value and drops the pending frame before a history replay lands', async () => {
    let flush: (() => void) | null = null;
    let cancelledFrames = 0;
    const harness = createHarness({
      schedulePreview: (callback) => {
        flush = callback;
        return () => {
          cancelledFrames += 1;
        };
      },
    });
    const { controller, document, history, historyController } = harness;
    expect(controller.commitPrepared('Rename', prepareRename(harness, 'Recorded'))).toEqual({ status: 'committed' });
    const session = controller.beginPreview()!;
    session.apply(rename('layer', 'Hovered'));
    flush!();
    expect(layerName(document())).toBe('Hovered');
    session.apply(rename('layer', 'Pending'));

    await expect(historyController.undo()).resolves.toEqual({ status: 'applied' });

    expect(cancelledFrames).toBe(1);
    expect(layerName(document())).toBe('Layer');
    expect(session.isActive()).toBe(false);
    expect(session.baseline()).toBeNull();
    // The widget's late cancel and commit change nothing: the undone step stays undone.
    session.cancel();
    expect(session.commit('Rename', prepareRename(harness, 'Late'))).toEqual({ status: 'busy' });
    expect(layerName(document())).toBe('Layer');
    expect(history.canUndo()).toBe(false);
    expect(history.canRedo()).toBe(true);
  });

  it('restores a previewed value before a commit from elsewhere or a newer session lands', () => {
    const { controller, document, history } = createHarness();
    const session = controller.beginPreview()!;
    session.apply(rename('layer', 'Hovered'));
    expect(layerName(document())).toBe('Hovered');

    expect(
      controller.commit(
        'Lock',
        { id: 'layer', patch: { isLocked: true }, type: 'updateCanvasLayer' },
        { id: 'layer', patch: { isLocked: false }, type: 'updateCanvasLayer' }
      )
    ).toEqual({ status: 'committed' });

    expect(getDocumentLayer(document(), 'layer')).toMatchObject({ isLocked: true, name: 'Layer' });
    expect(session.isActive()).toBe(false);

    const second = controller.beginPreview()!;
    second.apply({ id: 'layer', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' });
    expect(getDocumentLayer(document(), 'layer')?.opacity).toBe(0.5);
    const third = controller.beginPreview()!;

    expect(getDocumentLayer(document(), 'layer')?.opacity).toBe(1);
    expect(second.isActive()).toBe(false);
    expect(third.isActive()).toBe(true);
    expect(history.entries().past).toEqual(['Lock']);
  });

  it('holds each field as the gesture found it and previews one node per session', () => {
    const { controller, document } = createHarness();
    const session = controller.beginPreview()!;
    session.apply(rename('layer', 'A'));
    session.apply(rename('layer', 'B'));
    session.apply({ id: 'layer', patch: { opacity: 0.25 }, type: 'updateCanvasLayer' });

    expect(session.baseline()).toEqual({
      id: 'layer',
      patch: { name: 'Layer', opacity: 1 },
      type: 'updateCanvasLayer',
    });
    expect(session.apply(rename('missing', 'X'))).toBe(false);
    expect(
      session.apply({ config: { adjustments: [], layerType: 'raster' }, id: 'layer', type: 'updateCanvasLayerConfig' })
    ).toBe(false);

    session.cancel();
    expect(getDocumentLayer(document(), 'layer')).toMatchObject({ name: 'Layer', opacity: 1 });
    expect(session.isActive()).toBe(false);
  });

  it('refuses to preview a node the document does not hold', () => {
    const { controller, dispatched } = createHarness();
    const session = controller.beginPreview()!;

    expect(session.apply(rename('missing', 'X'))).toBe(false);
    expect(session.baseline()).toBeNull();
    expect(dispatched).toEqual([]);
  });
});

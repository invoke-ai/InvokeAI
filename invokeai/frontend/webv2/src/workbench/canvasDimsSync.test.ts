import { describe, expect, it } from 'vitest';

import type { CanvasProjectMutation } from './canvasProjectMutations';
import type { MainModelConfig } from './generation/types';
import type { WorkbenchState } from './types';
import type { WorkbenchAction } from './workbenchState';

import { type CanvasDimsSnapshot, createCanvasDimsSync, reconcileCanvasDims } from './canvasDimsSync';
import { getProjectWidgetValues } from './widgetState';
import { createWorkbenchStore } from './workbenchStore';

const bbox = (width: number, height: number, x = 0, y = 0) => ({ height, width, x, y });

const dispatchCanvas = (store: ReturnType<typeof createWorkbenchStore>, mutation: CanvasProjectMutation): void => {
  store.dispatch({ mutation, projectId: store.getState().activeProjectId, type: 'applyCanvasProjectMutation' });
};

const snapshot = (bboxW: number, bboxH: number, dimsW: number, dimsH: number): CanvasDimsSnapshot => ({
  bboxHeight: bboxH,
  bboxWidth: bboxW,
  dimsHeight: dimsH,
  dimsWidth: dimsW,
});

describe('reconcileCanvasDims (pure)', () => {
  it('no-ops when there is no canvas frame (not in canvas mode)', () => {
    expect(reconcileCanvasDims({ bbox: null, dims: { height: 512, width: 512 }, grid: 8, prev: null })).toEqual({
      kind: 'none',
    });
  });

  it('no-ops when the bbox and dims already agree', () => {
    expect(
      reconcileCanvasDims({
        bbox: bbox(1024, 1024),
        dims: { height: 1024, width: 1024 },
        grid: 8,
        prev: snapshot(1024, 1024, 1024, 1024),
      })
    ).toEqual({ kind: 'none' });
  });

  it('maps a bbox size change onto the generate dims and re-derives the aspect id', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(512, 768),
      dims: { height: 1024, width: 1024 },
      grid: 8,
      prev: snapshot(1024, 1024, 1024, 1024),
    });

    expect(result).toEqual({
      aspectRatioId: '2:3',
      aspectRatioValue: 512 / 768,
      height: 768,
      kind: 'patch-dims',
      width: 512,
    });
  });

  it('emits the bbox ratio as aspectRatioValue for a non-square bbox (not just the preset id)', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(1024, 768),
      dims: { height: 1024, width: 1024 },
      grid: 8,
      prev: snapshot(1024, 1024, 1024, 1024),
    });

    // 4:3 as an id, but the numeric ratio must agree so downstream constraint
    // math (GenerateDimensionFields' getActiveRatio) uses the fresh ratio
    // instead of a stale 1.0 left over from the previous (square) dims.
    expect(result).toMatchObject({ aspectRatioId: '4:3', aspectRatioValue: 1024 / 768, kind: 'patch-dims' });
  });

  it('re-derives a preset aspect id for a matching bbox ratio', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(1024, 768),
      dims: { height: 1024, width: 1024 },
      grid: 8,
      prev: snapshot(1024, 1024, 1024, 1024),
    });

    expect(result).toMatchObject({ aspectRatioId: '4:3', kind: 'patch-dims' });
  });

  it('re-derives Free for a non-preset bbox ratio', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(1000, 512),
      dims: { height: 1024, width: 1024 },
      grid: 8,
      prev: snapshot(1024, 1024, 1024, 1024),
    });

    expect(result).toMatchObject({ aspectRatioId: 'Free', kind: 'patch-dims' });
  });

  it('ignores a position-only bbox move (same width/height)', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(1024, 1024, 128, 64),
      dims: { height: 1024, width: 1024 },
      grid: 8,
      prev: snapshot(1024, 1024, 1024, 1024),
    });

    expect(result).toEqual({ kind: 'none' });
  });

  it('resizes the bbox to changed dims, keeping the top-left position', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(512, 768, 40, 24),
      dims: { height: 768, width: 768 },
      grid: 8,
      prev: snapshot(512, 768, 512, 768),
    });

    expect(result).toEqual({ bbox: { height: 768, width: 768, x: 40, y: 24 }, kind: 'set-bbox' });
  });

  it('snaps dims to the grid when resizing the bbox', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(512, 512),
      dims: { height: 100, width: 100 },
      grid: 16,
      prev: snapshot(512, 512, 512, 512),
    });

    expect(result).toEqual({ bbox: { height: 96, width: 96, x: 0, y: 0 }, kind: 'set-bbox' });
  });

  it('clamps a sub-grid dimension up to the grid minimum', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(512, 512),
      dims: { height: 1, width: 1 },
      grid: 16,
      prev: snapshot(512, 512, 512, 512),
    });

    expect(result).toEqual({ bbox: { height: 16, width: 16, x: 0, y: 0 }, kind: 'set-bbox' });
  });

  it('no-ops the dims->bbox direction when the snapped size already matches the bbox', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(96, 96),
      dims: { height: 100, width: 100 },
      grid: 16,
      prev: snapshot(96, 96, 96, 96),
    });

    expect(result).toEqual({ kind: 'none' });
  });

  it('lets the bbox win on first run (no prior snapshot)', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(512, 512),
      dims: { height: 256, width: 256 },
      grid: 8,
      prev: null,
    });

    expect(result).toEqual({ aspectRatioId: '1:1', aspectRatioValue: 1, height: 512, kind: 'patch-dims', width: 512 });
  });

  it('converges: applying each direction twice is a fixed point', () => {
    // bbox -> dims, then re-reconcile the resulting agreed state.
    const first = reconcileCanvasDims({
      bbox: bbox(640, 512),
      dims: { height: 1024, width: 1024 },
      grid: 8,
      prev: snapshot(1024, 1024, 1024, 1024),
    });
    expect(first.kind).toBe('patch-dims');
    const afterPatch = reconcileCanvasDims({
      bbox: bbox(640, 512),
      dims: { height: 512, width: 640 },
      grid: 8,
      prev: snapshot(640, 512, 640, 512),
    });
    expect(afterPatch).toEqual({ kind: 'none' });

    // dims -> bbox, then re-reconcile the resulting agreed state.
    const second = reconcileCanvasDims({
      bbox: bbox(512, 512, 10, 20),
      dims: { height: 512, width: 768 },
      grid: 8,
      prev: snapshot(512, 512, 512, 512),
    });
    expect(second.kind).toBe('set-bbox');
    const afterResize = reconcileCanvasDims({
      bbox: bbox(768, 512, 10, 20),
      dims: { height: 512, width: 768 },
      grid: 8,
      prev: snapshot(768, 512, 768, 512),
    });
    expect(afterResize).toEqual({ kind: 'none' });
  });
});

const model: MainModelConfig = { base: 'sdxl', key: 'test-model', name: 'Test Model', type: 'main' };

const getActiveGenerate = (state: WorkbenchState) => {
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);
  if (!project) {
    throw new Error('no active project');
  }
  return { bbox: project.canvas.document.bbox, values: getProjectWidgetValues(project, 'generate') };
};

/** A store that counts only the dispatches the sync itself issues. */
const setupCanvasStore = () => {
  const store = createWorkbenchStore();

  // Put the active project into canvas mode with concrete generate dims/model.
  store.dispatch({ type: 'setInvocationSource', sourceId: 'canvas' });
  store.dispatch({
    type: 'patchGenerateSettings',
    values: { height: 1024, model, modelKey: model.key, width: 1024 },
  });

  let syncDispatches = 0;
  const countingStore = {
    dispatch: (action: WorkbenchAction) => {
      syncDispatches += 1;
      store.dispatch(action);
    },
    getState: store.getState,
    subscribe: store.subscribe,
  };

  const sync = createCanvasDimsSync(countingStore);

  return { getSyncDispatches: () => syncDispatches, store, sync };
};

describe('createCanvasDimsSync (wiring)', () => {
  it('does not dispatch on mount when bbox and dims already agree', () => {
    const { getSyncDispatches, sync } = setupCanvasStore();

    expect(getSyncDispatches()).toBe(0);
    sync.dispose();
  });

  it('patches the generate dims when the bbox changes, without looping', () => {
    const { getSyncDispatches, store, sync } = setupCanvasStore();

    dispatchCanvas(store, { bbox: { height: 768, width: 512, x: 0, y: 0 }, type: 'setCanvasBbox' });

    const { values } = getActiveGenerate(store.getState());
    expect(values.width).toBe(512);
    expect(values.height).toBe(768);
    expect(values.aspectRatioId).toBe('2:3');
    // The numeric ratio must be re-derived alongside the id, or downstream
    // constraint math (GenerateDimensionFields' getActiveRatio, which prefers
    // aspectRatioValue whenever it is > 0) keeps constraining edits to the
    // stale ratio from before the bbox drag.
    expect(values.aspectRatioValue).toBeCloseTo(512 / 768);
    // Exactly one sync dispatch (the echo is a no-op, so no unbounded loop).
    expect(getSyncDispatches()).toBe(1);
    sync.dispose();
  });

  it('a bbox-driven ratio is what a subsequent constrained width edit uses (not the pre-drag ratio)', () => {
    const { store, sync } = setupCanvasStore();

    // Drag the bbox to a non-square 4:3 frame; dims should follow.
    dispatchCanvas(store, { bbox: { height: 768, width: 1024, x: 0, y: 0 }, type: 'setCanvasBbox' });
    const { values } = getActiveGenerate(store.getState());
    expect(values.aspectRatioId).toBe('4:3');
    expect(values.aspectRatioValue).toBeCloseTo(1024 / 768);

    const aspectRatioValue = values.aspectRatioValue as number;
    const width = values.width as number;
    const height = values.height as number;

    // Mirror GenerateDimensionFields' getActiveRatio: prefer aspectRatioValue
    // when positive, else fall back to the live width/height.
    const activeRatio = aspectRatioValue > 0 ? aspectRatioValue : width / height;
    const nextWidth = 800;
    const constrainedHeight = nextWidth / activeRatio;

    // Before the fix this would divide by the stale ratio (1.0, from the
    // original 1024x1024 dims) and yield the wrong height (800 instead of 600).
    expect(constrainedHeight).toBeCloseTo(600);
    sync.dispose();
  });

  it('resizes the bbox when the generate dims change, without looping', () => {
    const { getSyncDispatches, store, sync } = setupCanvasStore();

    // Move the frame first so the resize must preserve the top-left position.
    dispatchCanvas(store, { bbox: { height: 1024, width: 1024, x: 32, y: 48 }, type: 'setCanvasBbox' });
    const before = getSyncDispatches();

    store.dispatch({ type: 'patchGenerateSettings', values: { width: 512 } });

    const { bbox: nextBbox } = getActiveGenerate(store.getState());
    expect(nextBbox).toEqual({ height: 1024, width: 512, x: 32, y: 48 });
    expect(getSyncDispatches() - before).toBe(1);
    sync.dispose();
  });

  it('driving the dims off-grid still keeps the dispatch count bounded (no snap/reconcile loop)', () => {
    const { getSyncDispatches, store, sync } = setupCanvasStore();

    // Move the frame first so the resize must preserve the top-left position.
    dispatchCanvas(store, { bbox: { height: 1024, width: 1024, x: 32, y: 48 }, type: 'setCanvasBbox' });
    const before = getSyncDispatches();

    // 501 is not a multiple of the sdxl grid (8): dims -> bbox must snap it,
    // and the bbox -> dims echo from that snapped bbox must not re-trigger
    // another round of reconciliation (the snapped bbox and the still-501-wide
    // dims disagree by construction, but the sync's re-entrancy guard must
    // still hold the total dispatch count for this one external change to 1).
    store.dispatch({ type: 'patchGenerateSettings', values: { width: 501 } });

    const { bbox: nextBbox } = getActiveGenerate(store.getState());
    expect(nextBbox.width).toBe(504); // snapped up to the nearest multiple of 8
    expect(getSyncDispatches() - before).toBe(1);
    sync.dispose();
  });

  it('ignores a position-only bbox move', () => {
    const { getSyncDispatches, store, sync } = setupCanvasStore();

    dispatchCanvas(store, { bbox: { height: 1024, width: 1024, x: 100, y: 200 }, type: 'setCanvasBbox' });

    const { values } = getActiveGenerate(store.getState());
    expect(values.width).toBe(1024);
    expect(values.height).toBe(1024);
    expect(getSyncDispatches()).toBe(0);
    sync.dispose();
  });

  it('stays inert when the project is not invoking into the canvas', () => {
    const store = createWorkbenchStore();
    store.dispatch({ type: 'setInvocationSource', sourceId: 'generate' });
    store.dispatch({
      type: 'patchGenerateSettings',
      values: { height: 1024, model, modelKey: model.key, width: 1024 },
    });

    let syncDispatches = 0;
    const countingStore = {
      dispatch: (action: WorkbenchAction) => {
        syncDispatches += 1;
        store.dispatch(action);
      },
      getState: store.getState,
      subscribe: store.subscribe,
    };
    const sync = createCanvasDimsSync(countingStore);

    dispatchCanvas(store, { bbox: { height: 512, width: 512, x: 0, y: 0 }, type: 'setCanvasBbox' });

    const { values } = getActiveGenerate(store.getState());
    // Generate dims untouched: non-canvas behavior is exactly as today.
    expect(values.width).toBe(1024);
    expect(values.height).toBe(1024);
    expect(syncDispatches).toBe(0);
    sync.dispose();
  });

  it('aligns dims to the bbox when a project enters canvas mode', () => {
    const store = createWorkbenchStore();
    // Diverge the frame from the dims while still in generate mode.
    store.dispatch({
      type: 'patchGenerateSettings',
      values: { height: 1024, model, modelKey: model.key, width: 1024 },
    });
    dispatchCanvas(store, { bbox: { height: 640, width: 896, x: 0, y: 0 }, type: 'setCanvasBbox' });

    const sync = createCanvasDimsSync(store);
    // Switching into canvas mode should let the bbox win and drive the dims.
    store.dispatch({ type: 'setInvocationSource', sourceId: 'canvas' });

    const { values } = getActiveGenerate(store.getState());
    expect(values.width).toBe(896);
    expect(values.height).toBe(640);
    sync.dispose();
  });
});

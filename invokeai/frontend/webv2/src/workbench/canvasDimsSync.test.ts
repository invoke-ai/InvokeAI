import type { GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type { WorkbenchState } from '@workbench/projectContracts';

import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities.testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { CanvasProjectMutation } from './canvasProjectMutations';

import { type CanvasDimsSnapshot, createCanvasDimsSync, reconcileCanvasDims } from './canvasDimsSync';
import { getProjectWidgetValues } from './widgetState';
import { createWorkbenchStore } from './workbenchStore';

const bbox = (width: number, height: number, x = 0, y = 0) => ({ height, width, x, y });

const dispatchCanvas = (store: ReturnType<typeof createWorkbenchStore>, mutation: CanvasProjectMutation): void => {
  store.commands.canvas.apply(store.getState().activeProjectId, mutation);
};

const snapshot = (bboxW: number, bboxH: number, dimsW: number, dimsH: number, grid = 8): CanvasDimsSnapshot => ({
  bboxHeight: bboxH,
  bboxWidth: bboxW,
  dimsHeight: dimsH,
  dimsWidth: dimsW,
  grid,
});

describe('reconcileCanvasDims (pure)', () => {
  it('maps a free bbox size change onto the generate dims, leaving the ratio free', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(512, 768),
      dims: { height: 1024, width: 1024 },
      grid: 8,
      prev: snapshot(1024, 1024, 1024, 1024),
    });

    // Even though 512×768 is 2:3: a preset the person did not pick would lock the frame they left free.
    expect(result).toEqual({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: false,
      aspectRatioValue: 512 / 768,
      height: 768,
      kind: 'patch-dims',
      width: 512,
    });
  });

  it('keeps the person’s lock as it is for a frame of its ratio, and carries it to a frame of another', () => {
    const locked = { aspectRatioId: '4:3', aspectRatioIsLocked: true, aspectRatioValue: 4 / 3 } as const;
    expect(
      reconcileCanvasDims({
        aspect: locked,
        bbox: bbox(1024, 768),
        dims: { height: 1024, width: 1024 },
        grid: 8,
        prev: snapshot(1024, 1024, 1024, 1024),
      })
    ).toEqual({ height: 768, kind: 'patch-dims', width: 1024 });
    expect(
      reconcileCanvasDims({
        aspect: locked,
        bbox: bbox(1024, 576),
        dims: { height: 1024, width: 1024 },
        grid: 8,
        prev: snapshot(1024, 1024, 1024, 1024),
      })
    ).toEqual({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 1024 / 576,
      height: 576,
      kind: 'patch-dims',
      width: 1024,
    });
  });

  it.each([
    [
      'resizes the bbox to changed dims around its centre',
      { bbox: bbox(512, 768, 40, 24), dims: { height: 768, width: 768 }, grid: 8, prev: snapshot(512, 768, 512, 768) },
      { height: 768, width: 768, x: -88, y: 24 },
    ],
    [
      'snaps dims to the grid when resizing the bbox',
      { bbox: bbox(512, 512), dims: { height: 100, width: 100 }, grid: 16, prev: snapshot(512, 512, 512, 512, 16) },
      { height: 96, width: 96, x: 208, y: 208 },
    ],
    [
      'clamps a tiny dimension up to the shared generation minimum',
      { bbox: bbox(512, 512), dims: { height: 1, width: 1 }, grid: 16, prev: snapshot(512, 512, 512, 512, 16) },
      { height: 64, width: 64, x: 224, y: 224 },
    ],
  ])('%s', (_label, input, expected) => {
    expect(reconcileCanvasDims(input)).toEqual({ bbox: expected, kind: 'set-bbox' });
  });

  it('clamps an oversized bbox to the shared processing maximum without changing its footprint', () => {
    const result = reconcileCanvasDims({
      bbox: bbox(5000, 6000),
      dims: { height: 512, width: 512 },
      grid: 16,
      prev: snapshot(512, 512, 512, 512, 16),
    });

    expect(result).toEqual({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: false,
      aspectRatioValue: 5 / 6,
      height: 4096,
      kind: 'patch-dims',
      width: 4096,
    });
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

  store.commands.generation.setSource('canvas');
  store.commands.generation.patchSettings({ height: 1024, model, modelKey: model.key, width: 1024 });

  let syncDispatches = 0;
  const countingStore = {
    commands: {
      canvas: {
        apply: (...args: Parameters<typeof store.commands.canvas.apply>) => {
          syncDispatches += 1;
          return store.commands.canvas.apply(...args);
        },
        appendStagingCandidate: store.commands.canvas.appendStagingCandidate,
      },
      generation: {
        ...store.commands.generation,
        patchSettings: (...args: Parameters<typeof store.commands.generation.patchSettings>) => {
          syncDispatches += 1;
          store.commands.generation.patchSettings(...args);
        },
      },
      widgets: store.commands.widgets,
    },
    getState: store.getState,
    subscribe: store.subscribe,
  };

  const sync = createCanvasDimsSync(countingStore);

  return { getSyncDispatches: () => syncDispatches, store, sync };
};

describe('Generate size edits on the canvas (wiring)', () => {
  seedArchitectureCapabilities();

  const SQUARE_RESET = {
    aspectRatioId: '1:1',
    aspectRatioIsLocked: true,
    aspectRatioValue: 1,
    height: 1024,
    width: 1024,
  } as const;

  it('resizes the canvas frame around its centre, then settles', () => {
    const { store, sync } = setupCanvasStore();
    dispatchCanvas(store, { bbox: { height: 1216, width: 832, x: 100, y: 100 }, type: 'setCanvasBbox' });
    const apply = vi.spyOn(store.commands.canvas, 'apply');

    store.commands.generation.patchSettings(SQUARE_RESET);

    const { bbox: nextBbox, values } = getActiveGenerate(store.getState());
    expect(nextBbox).toEqual({ height: 1024, width: 1024, x: 4, y: 196 });
    expect(values).toMatchObject(SQUARE_RESET);
    expect(store.getState().projects[0]?.invocation.sourceId).toBe('canvas');
    // No canvas engine is open, so the frame write is the sync's own.
    expect(apply.mock.calls.map((call) => call[2])).toEqual(['system']);

    // Frame and size now agree, so the next reconciliation writes nothing.
    apply.mockClear();
    store.commands.generation.patchSettings({ steps: 31 });
    expect(apply).not.toHaveBeenCalled();

    // A later size edit keeps the centre too.
    store.commands.generation.patchSettings({ height: 1024, width: 512 });
    expect(getActiveGenerate(store.getState()).bbox).toEqual({ height: 1024, width: 512, x: 260, y: 196 });
    sync.dispose();
  });

  it('records the frame resize as a canvas step while the canvas is open, so canvas undo can reverse it', () => {
    const store = createWorkbenchStore();
    store.commands.generation.setSource('canvas');
    store.commands.generation.patchSettings({ height: 1024, model, modelKey: model.key, width: 1024 });
    const commits: { from: unknown; to: unknown }[] = [];
    const sync = createCanvasDimsSync(store, (projectId, from, to) => {
      commits.push({ from, to });
      // Stands in for the engine's commitStructural, which dispatches the same mutation as a user edit.
      return store.commands.canvas.apply(projectId, { bbox: to, type: 'setCanvasBbox' });
    });
    const apply = vi.spyOn(store.commands.canvas, 'apply');

    store.commands.generation.patchSettings({ width: 512 });

    expect(commits).toEqual([
      { from: { height: 1024, width: 1024, x: 0, y: 0 }, to: { height: 1024, width: 512, x: 256, y: 0 } },
    ]);
    // Only the commit's own write; no system write follows it.
    expect(apply.mock.calls.map((call) => call[2])).toEqual([undefined]);
    expect(getActiveGenerate(store.getState()).bbox).toEqual({ height: 1024, width: 512, x: 256, y: 0 });
    sync.dispose();
  });
});

describe('createCanvasDimsSync (wiring)', () => {
  seedArchitectureCapabilities();

  it('patches the generate dims when the bbox changes, without looping', () => {
    const { getSyncDispatches, store, sync } = setupCanvasStore();

    dispatchCanvas(store, { bbox: { height: 768, width: 512, x: 0, y: 0 }, type: 'setCanvasBbox' });

    const { values } = getActiveGenerate(store.getState());
    expect(values.width).toBe(512);
    expect(values.height).toBe(768);
    expect(values.aspectRatioId).toBe('Free');
    // Re-derive numeric aspectRatioValue because downstream constraints prefer it over the id.
    expect(values.aspectRatioValue).toBeCloseTo(512 / 768);
    // Exactly one sync dispatch (the echo is a no-op, so no unbounded loop).
    expect(getSyncDispatches()).toBe(1);
    sync.dispose();
  });

  it('snaps an off-grid bbox and generation dimensions to the active model grid', () => {
    const { getSyncDispatches, store, sync } = setupCanvasStore();

    dispatchCanvas(store, { bbox: { height: 503, width: 501, x: 32, y: 48 }, type: 'setCanvasBbox' });

    const { bbox: nextBbox, values } = getActiveGenerate(store.getState());
    expect(nextBbox).toEqual({ height: 503, width: 501, x: 32, y: 48 });
    expect(values.width).toBe(504);
    expect(values.height).toBe(504);
    expect(getSyncDispatches()).toBe(1);
    sync.dispose();
  });

  it('resizes the bbox when the generate dims change, without looping', () => {
    const { getSyncDispatches, store, sync } = setupCanvasStore();

    // Move the frame first so the resize must keep its centre.
    dispatchCanvas(store, { bbox: { height: 1024, width: 1024, x: 32, y: 48 }, type: 'setCanvasBbox' });
    const before = getSyncDispatches();

    store.commands.generation.patchSettings({ width: 512 });

    const { bbox: nextBbox } = getActiveGenerate(store.getState());
    expect(nextBbox).toEqual({ height: 1024, width: 512, x: 288, y: 48 });
    expect(getSyncDispatches() - before).toBe(1);
    sync.dispose();
  });

  it('driving the dims off-grid still keeps the dispatch count bounded (no snap/reconcile loop)', () => {
    const { getSyncDispatches, store, sync } = setupCanvasStore();

    // Move the frame first so the resize must keep its centre.
    dispatchCanvas(store, { bbox: { height: 1024, width: 1024, x: 32, y: 48 }, type: 'setCanvasBbox' });
    const before = getSyncDispatches();

    // Off-grid input requires a bbox snap and dimension echo; re-entry must keep those writes bounded.
    store.commands.generation.patchSettings({ width: 501 });

    const { bbox: nextBbox, values } = getActiveGenerate(store.getState());
    expect(nextBbox.width).toBe(504); // snapped up to the nearest multiple of 8
    expect(values.width).toBe(504);
    expect(getSyncDispatches() - before).toBe(2);
    sync.dispose();
  });

  it('stays inert when the project is not invoking into the canvas', () => {
    const store = createWorkbenchStore();
    store.commands.generation.setSource('generate');
    // Lock the source: a bbox drag is itself a canvas intent signal and would otherwise auto-route the project into
    // canvas mode.
    store.commands.generation.toggleSourceLock();
    store.commands.generation.patchSettings({ height: 1024, model, modelKey: model.key, width: 1024 });

    let syncDispatches = 0;
    const countingStore = {
      commands: {
        canvas: {
          apply: (...args: Parameters<typeof store.commands.canvas.apply>) => {
            syncDispatches += 1;
            return store.commands.canvas.apply(...args);
          },
          appendStagingCandidate: store.commands.canvas.appendStagingCandidate,
        },
        generation: {
          ...store.commands.generation,
          patchSettings: (...args: Parameters<typeof store.commands.generation.patchSettings>) => {
            syncDispatches += 1;
            store.commands.generation.patchSettings(...args);
          },
        },
        widgets: store.commands.widgets,
      },
      getState: store.getState,
      subscribe: store.subscribe,
    };
    const sync = createCanvasDimsSync(countingStore);

    dispatchCanvas(store, { bbox: { height: 512, width: 512, x: 0, y: 0 }, type: 'setCanvasBbox' });

    const { values } = getActiveGenerate(store.getState());
    expect(values.width).toBe(1024);
    expect(values.height).toBe(1024);
    expect(syncDispatches).toBe(0);
    sync.dispose();
  });
});

describe('createCanvasDimsSync before the capability table arrives', () => {
  afterEach(resetArchitectureCapabilities);

  const wanModel: MainModelConfig = { base: 'wan', key: 'test-wan-model', name: 'Test Wan Model', type: 'main' };

  /** A saved canvas project whose frame is 8-aligned but not on Wan's 16px grid. */
  const setupWanProject = () => {
    const store = createWorkbenchStore();

    store.commands.generation.setSource('canvas');
    store.commands.generation.patchSettings({
      height: 1024,
      model: wanModel,
      modelKey: wanModel.key,
      width: 1024,
    });
    dispatchCanvas(store, { bbox: { height: 1032, width: 1032, x: 0, y: 0 }, type: 'setCanvasBbox' });

    let syncDispatches = 0;
    const countingStore = {
      commands: {
        canvas: store.commands.canvas,
        generation: {
          ...store.commands.generation,
          patchSettings: (...args: Parameters<typeof store.commands.generation.patchSettings>) => {
            syncDispatches += 1;
            store.commands.generation.patchSettings(...args);
          },
        },
        widgets: store.commands.widgets,
      },
      getState: store.getState,
      subscribe: store.subscribe,
    };

    return { getSyncDispatches: () => syncDispatches, store, sync: createCanvasDimsSync(countingStore) };
  };

  it('reconciles an external generator at once, which will never get a row to wait for', () => {
    // External providers never receive a capability row, so waiting would disable sync permanently.
    const externalModel = {
      base: 'external',
      capabilities: { modes: ['txt2img'], supports_seed: true },
      format: 'external_api',
      key: 'external-model',
      name: 'OpenAI Image',
      provider_id: 'openai',
      type: 'external_image_generator',
    } as GenerateWidgetValues['model'];
    const store = createWorkbenchStore();

    store.commands.generation.setSource('canvas');
    store.commands.generation.patchSettings({
      height: 1024,
      model: externalModel,
      modelKey: externalModel.key,
      width: 1024,
    });
    dispatchCanvas(store, { bbox: { height: 1032, width: 1032, x: 0, y: 0 }, type: 'setCanvasBbox' });

    const sync = createCanvasDimsSync(store);

    expect(getActiveGenerate(store.getState()).values.width).toBe(1032);
    sync.dispose();
  });

  it('reconciles as soon as the table lands, with no other change to the project', () => {
    // Capability arrival alone must retrigger sync; an otherwise consistent project emits no workbench change.
    const { getSyncDispatches, store, sync } = setupWanProject();

    setArchitectureCapabilities(architectureCapabilitiesFixture);

    // The 1032px frame snapped onto Wan's declared 16px grid, not the fallback's 8.
    const { values } = getActiveGenerate(store.getState());
    expect(values.width).toBe(1040);
    expect(values.height).toBe(1040);
    expect(getSyncDispatches()).toBe(1);
    sync.dispose();
  });
});

describe('createCanvasDimsSync (manual render size)', () => {
  seedArchitectureCapabilities();

  const getRenderSize = (store: ReturnType<typeof createWorkbenchStore>) => {
    const state = store.getState();
    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);
    const { scaledHeight, scaledWidth } = project ? getProjectWidgetValues(project, 'canvas') : {};
    return { height: scaledHeight, width: scaledWidth };
  };

  /** A canvas project whose frame is locked to 3:2 and renders at a custom 1536x1024 (1,572,864 px). */
  const setupLockedCustom = () => {
    const store = createWorkbenchStore();
    store.commands.generation.setSource('canvas');
    dispatchCanvas(store, { bbox: { height: 832, width: 1248, x: 0, y: 0 }, type: 'setCanvasBbox' });
    store.commands.generation.patchSettings({
      aspectRatioId: '3:2',
      aspectRatioIsLocked: true,
      aspectRatioValue: 3 / 2,
      height: 832,
      model,
      modelKey: model.key,
      width: 1248,
    });
    const sync = createCanvasDimsSync(store);
    store.commands.widgets.patchValues('canvas', { scaleMethod: 'manual', scaledHeight: 1024, scaledWidth: 1536 });
    return { store, sync };
  };

  it('reshapes the custom size to a newly chosen ratio at about the same area', () => {
    const { store, sync } = setupLockedCustom();

    store.commands.generation.patchSettings({
      aspectRatioId: '16:9',
      aspectRatioValue: 16 / 9,
      height: 768,
      width: 1360,
    });

    // sqrt(1,572,864 × 16/9) ≈ 1672.2 wide and 940.6 tall, each to the nearest multiple of 8.
    expect(getRenderSize(store)).toEqual({ height: 944, width: 1672 });
    sync.dispose();
  });

  it('follows a bbox resize while the ratio is locked', () => {
    const { store, sync } = setupLockedCustom();

    dispatchCanvas(store, { bbox: { height: 700, width: 1000, x: 0, y: 0 }, type: 'setCanvasBbox' });

    // The bbox's 10:7 at the same area: ≈ 1499.0 × 1049.3.
    expect(getRenderSize(store)).toEqual({ height: 1048, width: 1496 });
    sync.dispose();
  });
});

import type { CanvasImageUploadResult } from '@workbench/canvas-engine/backend/canvasImages';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type {
  GenerateModelConfig,
  GenerateReferenceImage,
  GenerateReferenceImageAsset,
  MainModelConfig,
} from '@workbench/generation/types';
import type {
  CanvasDocumentContractV2,
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  WorkbenchState,
} from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { createCompositeDedupeCache } from '@workbench/canvas-engine/export/compositeForGeneration';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { getDefaultGenerateSettings } from '@workbench/generation/baseGenerationPolicies';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { RunCanvasInvocationDeps } from './prepareCanvasInvocation';

import { DEFAULT_CANVAS_COMPOSITING } from './canvasCompositing';
import { resolveRegionalReferenceImages, runCanvasInvocation } from './prepareCanvasInvocation';

const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1-model', name: 'SD 1.5', type: 'main' };
const externalModel: GenerateModelConfig = {
  base: 'external',
  capabilities: { modes: ['txt2img'], supports_seed: true },
  format: 'external_api',
  key: 'external-model',
  name: 'OpenAI Image',
  provider_id: 'openai',
  type: 'external_image_generator',
};

const generateValuesFor = (model: GenerateModelConfig): Record<string, unknown> => ({
  ...getDefaultGenerateSettings(model),
  model,
  modelKey: model.key,
  positivePrompt: 'a canvas prompt',
  seed: 7,
  shouldRandomizeSeed: false,
});

const rasterLayer = (id: string, size = 64): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: size, imageName: `${id}.png`, width: size }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const makeDoc = (layers: CanvasRasterLayerContractV2[], size = 64): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: size, width: size, x: 0, y: 0 },
  height: size,
  layers,
  selectedLayerId: null,
  version: 2,
  width: size,
});

/** Uniform-alpha ImageData (255 = fully opaque → bboxFullyCovered true). */
const uniformImageData = (width: number, height: number, alpha: number): ImageData => {
  const data = new Uint8ClampedArray(Math.max(1, width) * Math.max(1, height) * 4);
  for (let i = 3; i < data.length; i += 4) {
    data[i] = alpha;
  }
  return { colorSpace: 'srgb', data, height, width } as unknown as ImageData;
};

interface Harness {
  deps: RunCanvasInvocationDeps;
  dispatch: ReturnType<typeof vi.fn>;
  uploadImage: ReturnType<typeof vi.fn>;
  flushPendingUploads: ReturnType<typeof vi.fn>;
  events: string[];
  submittedGraphs: () => Extract<WorkbenchAction, { type: 'submitCanvasInvocationSnapshot' }>[];
  notices: () => { message?: string }[];
}

interface HarnessOptions {
  document?: CanvasDocumentContractV2;
  destination?: 'canvas' | 'gallery';
  model?: GenerateModelConfig;
  strength?: number;
  alpha?: number;
  inFlight?: Set<string>;
  flushPendingUploads?: () => Promise<void>;
  uploadImage?: (blob: Blob) => Promise<CanvasImageUploadResult>;
  projectId?: string;
  dispatch?: ReturnType<typeof vi.fn<(action: WorkbenchAction) => void>>;
  models?: RunCanvasInvocationDeps['models'];
}

const makeHarness = (options: HarnessOptions = {}): Harness => {
  const stub = createTestStubRasterBackend();
  const events: string[] = [];
  const model = options.model ?? sd1Model;
  const alpha = options.alpha ?? 255;

  const backend = {
    createSurface: (w: number, h: number): RasterSurface => stub.createSurface(w, h),
    encodeSurface: (surface: RasterSurface): Promise<Blob> => stub.encodeSurface(surface),
  };

  const layerSurfaces = new Map<string, RasterSurface>();
  const getLayerSurface = (layerId: string): Promise<{ surface: RasterSurface; rect: Rect }> => {
    events.push('getLayerSurface');
    let surface = layerSurfaces.get(layerId);
    if (!surface) {
      surface = stub.createSurface(64, 64);
      layerSurfaces.set(layerId, surface);
    }
    // Content-sized: the layer's cache occupies its content rect (origin-anchored
    // for these 64×64 image layers).
    return Promise.resolve({ rect: { height: 64, width: 64, x: 0, y: 0 }, surface });
  };

  let counter = 0;
  const uploadImage = vi.fn(
    options.uploadImage ??
      ((blob: Blob): Promise<CanvasImageUploadResult> => {
        void blob;
        events.push('upload');
        counter += 1;
        return Promise.resolve({ height: 64, imageName: `composite-${counter}.png`, width: 64 });
      })
  );

  const flushPendingUploads = vi.fn(
    options.flushPendingUploads ??
      ((): Promise<void> =>
        new Promise((resolve) => {
          // Resolve on a microtask so a missing `await` would let compositing
          // start first — the ordering assertion depends on this.
          queueMicrotask(() => {
            events.push('flush');
            resolve();
          });
        }))
  );

  const dispatch = options.dispatch ?? vi.fn((_action: WorkbenchAction) => {});

  const deps: RunCanvasInvocationDeps = {
    compositing: DEFAULT_CANVAS_COMPOSITING,
    dedupe: createCompositeDedupeCache(),
    destination: options.destination ?? 'canvas',
    dispatch,
    executorDeps: {
      backend,
      getLayerSurface,
      hashBlob: (blob: Blob) => blob.text(),
      readImageData: (_surface, rect) => uniformImageData(rect.width, rect.height, alpha),
      uploadImage: uploadImage as (blob: Blob) => Promise<CanvasImageUploadResult>,
    },
    flushPendingUploads: flushPendingUploads as () => Promise<void>,
    generateValues: generateValuesFor(model),
    getDocument: () => options.document ?? makeDoc([rasterLayer('layer-a')]),
    inFlight: options.inFlight ?? new Set<string>(),
    models: options.models,
    projectId: options.projectId ?? 'project-1',
    projectSettings: { useCpuNoise: true },
    strength: options.strength ?? 0.75,
  };

  const submittedGraphs = () =>
    dispatch.mock.calls.map(([action]) => action).filter((action) => action.type === 'submitCanvasInvocationSnapshot');
  const notices = () =>
    dispatch.mock.calls.map(([action]) => action).filter((action) => action.type === 'recordNotice');

  return { deps, dispatch, events, flushPendingUploads, notices, submittedGraphs, uploadImage };
};

const controlLayer = (
  id: string,
  overrides: Partial<CanvasControlLayerContract['adapter']> = {},
  hasContent = true
): CanvasControlLayerContract => ({
  adapter: {
    beginEndStepPct: [0, 1],
    controlMode: 'balanced',
    kind: 'controlnet',
    model: null,
    weight: 0.75,
    ...overrides,
  },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: hasContent
    ? { image: { height: 64, imageName: `${id}.png`, width: 64 }, type: 'image' }
    : { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

describe('runCanvasInvocation', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('generates txt2img without any composite upload when content does not overlap the bbox', async () => {
    // Empty document → no raster content → txt2img, no executor work.
    const harness = makeHarness({ document: makeDoc([]) });

    await runCanvasInvocation(harness.deps);

    expect(harness.uploadImage).not.toHaveBeenCalled();
    expect(harness.events).not.toContain('getLayerSurface');

    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.graph.label).toBe('SD 1.5 txt2img');
    // Pure txt2img graph has no image-to-latents encode node.
    expect(submitted[0]!.graph.backendGraph!.nodes.canvas_i2l).toBeUndefined();
    expect(harness.notices()).toHaveLength(0);
  });

  it('generates img2img with the composite reference and the strength-derived denoising_start', async () => {
    const harness = makeHarness({ strength: 0.75 });

    await runCanvasInvocation(harness.deps);

    expect(harness.uploadImage).toHaveBeenCalledTimes(1);

    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.graph.label).toBe('SD 1.5 img2img');

    const nodes = submitted[0]!.graph.backendGraph!.nodes;
    const encode = Object.values(nodes).find((node: any) => node.type === 'i2l') as any;
    expect(encode?.image).toEqual({ image_name: 'composite-1.png' });
    // sd-1 is linear: denoising_start = 1 - 0.75.
    expect(nodes.denoise_latents.denoising_start).toBeCloseTo(0.25, 10);
    expect(harness.notices()).toHaveLength(0);
  });

  it('dispatches resolved prompt and seed metadata with the compiled canvas graph', async () => {
    const harness = makeHarness();
    harness.deps.generateValues = {
      ...harness.deps.generateValues,
      negativePrompt: 'low quality',
      positivePrompt: 'a resolved canvas prompt',
      seed: 123,
      shouldRandomizeSeed: false,
    };

    await runCanvasInvocation(harness.deps);

    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.generate).toMatchObject({
      negativePromptNodeId: 'negative_prompt',
      positivePromptNodeId: 'positive_prompt',
      seedNodeId: 'seed',
      values: {
        negativePrompt: 'low quality',
        positivePrompt: 'a resolved canvas prompt',
        seed: 123,
        shouldRandomizeSeed: false,
      },
    });
  });

  it('threads a Gallery destination through to the snapshot and a durable (non-intermediate) output', async () => {
    // Empty document → txt2img, so no composite upload noise; we only care about
    // the destination + is_intermediate wiring.
    const harness = makeHarness({ destination: 'gallery', document: makeDoc([]) });

    await runCanvasInvocation(harness.deps);

    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.destination).toBe('gallery');
    // A Gallery destination must produce a durable image, not a staging intermediate.
    expect(submitted[0]!.graph.backendGraph!.nodes.canvas_output!.is_intermediate).toBe(false);
  });

  it('marks the output intermediate for a Canvas destination (staging)', async () => {
    const harness = makeHarness({ destination: 'canvas', document: makeDoc([]) });

    await runCanvasInvocation(harness.deps);

    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.destination).toBe('canvas');
    expect(submitted[0]!.graph.backendGraph!.nodes.canvas_output!.is_intermediate).toBe(true);
  });

  it('awaits the upload flush before compositing', async () => {
    const harness = makeHarness();

    await runCanvasInvocation(harness.deps);

    // Flush must settle before any layer is rasterized/uploaded.
    expect(harness.events[0]).toBe('flush');
    expect(harness.events.indexOf('flush')).toBeLessThan(harness.events.indexOf('getLayerSurface'));
    expect(harness.events.indexOf('flush')).toBeLessThan(harness.events.indexOf('upload'));
  });

  it('plans from the POST-flush document (re-reads getDocument after the flush barrier)', async () => {
    // The flush dispatches `updateCanvasLayerSource` for just-persisted paint
    // layers, so the pre-flush snapshot references stale/empty sources. The
    // orchestrator must re-read the document after the flush and composite from
    // it — otherwise the stale refs build wrong dedupe keys and the empty-paint
    // filter reads a `null` bitmap that the flush already replaced.
    const preDoc = makeDoc([rasterLayer('pre')]);
    const postDoc = makeDoc([rasterLayer('post')]);
    let flushed = false;
    const harness = makeHarness({
      flushPendingUploads: () =>
        new Promise((resolve) => {
          queueMicrotask(() => {
            flushed = true;
            resolve();
          });
        }),
    });
    // Return the stale doc until the flush resolves, the fresh doc afterwards.
    harness.deps.getDocument = () => (flushed ? postDoc : preDoc);
    const surfaceIds: string[] = [];
    const inner = harness.deps.executorDeps.getLayerSurface;
    harness.deps.executorDeps.getLayerSurface = (layerId: string) => {
      surfaceIds.push(layerId);
      return inner(layerId);
    };

    await runCanvasInvocation(harness.deps);

    // The composite rasterized the post-flush layer, never the stale pre-flush one.
    expect(surfaceIds).toContain('post');
    expect(surfaceIds).not.toContain('pre');
  });

  it('records a notice and dispatches no snapshot when the graph fails validation', async () => {
    // External image generators are rejected by the graph compiler.
    const harness = makeHarness({ document: makeDoc([]), model: externalModel });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    const notices = harness.notices();
    expect(notices).toHaveLength(1);
    expect(notices[0]?.message).toContain('does not support canvas generation');
  });

  it('records a notice and dispatches no snapshot when the composite upload fails', async () => {
    const uploadImage = vi.fn(() => Promise.reject(new Error('upload exploded')));
    const harness = makeHarness({ uploadImage });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    const notices = harness.notices();
    expect(notices).toHaveLength(1);
    expect(notices[0]?.message).toBe('upload exploded');
  });

  it('blocks invocation with the shared reason code when a nonempty control layer has no model', async () => {
    const harness = makeHarness({ document: docWithLayers([controlLayer('control')]) });
    await runCanvasInvocation(harness.deps);
    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices()).toHaveLength(1);
    expect(harness.notices()[0]?.message).toContain('missing_model');
  });

  it('blocks invocation when a nonempty control layer has malformed numeric settings', async () => {
    const model = {
      base: 'sd-1',
      file_size: 1,
      format: 'checkpoint',
      hash: 'hash',
      key: 'controlnet',
      name: 'ControlNet',
      path: 'controlnet',
      source: 'controlnet',
      source_type: 'path' as const,
      type: 'controlnet',
    };
    const harness = makeHarness({
      document: docWithLayers([controlLayer('control', { model: model.key, weight: Number.NaN })]),
      models: [model],
    });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices()[0]?.message).toContain('invalid_adapter_values');
  });

  it('ignores an empty control layer with no model', async () => {
    const harness = makeHarness({ document: docWithLayers([controlLayer('control', {}, false)]) });
    await runCanvasInvocation(harness.deps);
    expect(harness.submittedGraphs()).toHaveLength(1);
    expect(harness.notices()).toHaveLength(0);
  });

  it('ignores a disabled nonempty control layer with no model', async () => {
    const disabled = { ...controlLayer('control'), isEnabled: false };
    const harness = makeHarness({ document: docWithLayers([disabled]) });
    await runCanvasInvocation(harness.deps);
    expect(harness.submittedGraphs()).toHaveLength(1);
    expect(harness.notices()).toHaveLength(0);
  });

  it('blocks the second enabled nonempty Control LoRA', async () => {
    const model = {
      base: 'flux',
      file_size: 1,
      format: 'checkpoint',
      hash: 'hash',
      key: 'control-lora',
      name: 'Control LoRA',
      path: 'control-lora',
      source: 'control-lora',
      source_type: 'path' as const,
      type: 'control_lora',
    };
    const flux = { ...sd1Model, base: 'flux' as const, key: 'flux', name: 'FLUX' };
    const harness = makeHarness({
      document: docWithLayers([
        controlLayer('first', { kind: 'control_lora', model: model.key }),
        controlLayer('second', { kind: 'control_lora', model: model.key }),
      ]),
      model: flux,
      models: [model],
    });
    await runCanvasInvocation(harness.deps);
    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices()[0]?.message).toContain('control_lora_limit');
  });

  it('prepares a Z-Image control with exact adapter and model configuration', async () => {
    const zImage = {
      ...sd1Model,
      base: 'z-image' as const,
      format: 'diffusers' as const,
      key: 'z-image',
      name: 'Z-Image',
    };
    const controlModel = {
      base: 'z-image',
      file_size: 1,
      format: 'checkpoint',
      hash: 'z-control-hash',
      key: 'z-control',
      name: 'Z Control',
      path: 'z-control',
      source: 'z-control',
      source_type: 'path' as const,
      type: 'controlnet',
    };
    const harness = makeHarness({
      document: docWithLayers([
        controlLayer('z-layer', {
          beginEndStepPct: [0.15, 0.85],
          controlMode: null,
          kind: 'z_image_control',
          model: controlModel.key,
          weight: 0.7,
        }),
      ]),
      model: zImage,
      models: [controlModel],
    });

    await runCanvasInvocation(harness.deps);

    expect(harness.notices()).toHaveLength(0);
    const node = harness.submittedGraphs()[0]?.graph.backendGraph?.nodes['z_image_control_z-layer'];
    expect(node).toMatchObject({
      begin_step_percent: 0.15,
      control_context_scale: 0.7,
      control_model: {
        base: 'z-image',
        hash: 'z-control-hash',
        key: 'z-control',
        name: 'Z Control',
        type: 'controlnet',
      },
      end_step_percent: 0.85,
      type: 'z_image_control',
    });
  });

  it('blocks a second enabled nonempty Z-Image control', async () => {
    const zImage = {
      ...sd1Model,
      base: 'z-image' as const,
      format: 'diffusers' as const,
      key: 'z-image',
      name: 'Z-Image',
    };
    const controlModel = {
      base: 'z-image',
      file_size: 1,
      format: 'checkpoint',
      hash: 'hash',
      key: 'z-control',
      name: 'Z Control',
      path: 'z-control',
      source: 'z-control',
      source_type: 'path' as const,
      type: 'controlnet',
    };
    const harness = makeHarness({
      document: docWithLayers([
        controlLayer('first', { kind: 'z_image_control', model: controlModel.key }),
        controlLayer('second', { kind: 'z_image_control', model: controlModel.key }),
      ]),
      model: zImage,
      models: [controlModel],
    });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices()[0]?.message).toContain('z_image_control_limit');
  });

  it('ignores a concurrent invoke while a prior prepare for the same project is in flight', async () => {
    const inFlight = new Set<string>();
    let releaseFlush = (): void => {};
    const gatedFlush = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          releaseFlush = resolve;
        })
    );

    const first = makeHarness({ flushPendingUploads: gatedFlush, inFlight });
    // Kick off the first invoke; it parks on the gated flush.
    const firstRun = runCanvasInvocation(first.deps);
    await Promise.resolve();

    // A second invoke for the same project (shares the in-flight set) is dropped.
    const second = makeHarness({ inFlight });
    await runCanvasInvocation(second.deps);

    expect(second.submittedGraphs()).toHaveLength(0);
    expect(second.notices()).toHaveLength(0);
    expect(second.uploadImage).not.toHaveBeenCalled();

    // Let the first invoke finish; it still submits exactly once.
    releaseFlush();
    await firstRun;
    expect(first.submittedGraphs()).toHaveLength(1);
  });

  it('enqueues into the originating project, not the active one, when the active project changes mid-flight', async () => {
    // Two real projects, driven through the real reducer, so we can tell
    // "landed in project A" apart from "landed in project B".
    let state: WorkbenchState = createInitialWorkbenchState();
    const originatingProjectId = state.activeProjectId;
    state = workbenchReducer(state, { type: 'createProject' });
    const otherProjectId = state.activeProjectId;
    expect(otherProjectId).not.toBe(originatingProjectId);

    let releaseFlush = (): void => {};
    const gatedFlush = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          releaseFlush = resolve;
        })
    );
    const dispatch = vi.fn((action: WorkbenchAction) => {
      state = workbenchReducer(state, action);
    });

    // The invoke is prepared while `originatingProjectId` is active...
    const harness = makeHarness({ dispatch, flushPendingUploads: gatedFlush, projectId: originatingProjectId });
    const invokePromise = runCanvasInvocation(harness.deps);
    await Promise.resolve();

    // ...but by the time the flush/composite/compile settles, the user has
    // switched to another project.
    expect(state.activeProjectId).toBe(otherProjectId);

    releaseFlush();
    await invokePromise;

    const originatingProject = state.projects.find((project) => project.id === originatingProjectId);
    const otherProject = state.projects.find((project) => project.id === otherProjectId);

    expect(originatingProject?.queue.items).toHaveLength(1);
    expect(originatingProject?.queue.items[0]?.snapshot.sourceId).toBe('canvas');
    expect(otherProject?.queue.items).toHaveLength(0);
  });
});

// ---- Inpaint / outpaint mode dispatch -------------------------------------

const inpaintMaskLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: { height: 64, imageName: `${id}-bmp`, width: 64 }, fill: { color: '#ff0000', style: 'solid' } },
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const docWithLayers = (layers: CanvasLayerContract[], size = 64): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: size, width: size, x: 0, y: 0 },
  height: size,
  layers,
  selectedLayerId: null,
  version: 2,
  width: size,
});

describe('runCanvasInvocation — inpaint / outpaint dispatch', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('dispatches an inpaint graph when content covers the bbox and a mask has content', async () => {
    const doc = docWithLayers([rasterLayer('base'), inpaintMaskLayer('mask')]);
    const harness = makeHarness({ document: doc, alpha: 255 });

    await runCanvasInvocation(harness.deps);

    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.graph.label).toBe('SD 1.5 inpaint');
    const nodes = submitted[0]!.graph.backendGraph!.nodes;
    expect(nodes.create_gradient_mask?.type).toBe('create_gradient_mask');
    expect(nodes.canvas_output?.type).toBe('invokeai_img_blend');
    // The gradient mask consumes the uploaded grayscale mask (base upload is composite-1).
    expect((nodes.create_gradient_mask?.mask as { image_name?: string } | undefined)?.image_name).toBeDefined();
    expect(harness.notices()).toHaveLength(0);
    // The mask layer was rasterized for the grayscale composite.
    expect(harness.events.filter((e) => e === 'getLayerSurface')).not.toHaveLength(0);
  });

  it('dispatches an outpaint graph when content only partially covers the bbox', async () => {
    // Partial alpha → bboxFullyCovered false → outpaint.
    const harness = makeHarness({ document: makeDoc([rasterLayer('base')]), alpha: 200 });

    await runCanvasInvocation(harness.deps);

    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.graph.label).toBe('SD 1.5 outpaint');
    const nodes = submitted[0]!.graph.backendGraph!.nodes;
    expect(nodes.infill?.type).toBe('infill_lama');
    expect(nodes.image_alpha_to_mask?.type).toBe('tomask');
    expect(nodes.canvas_output?.type).toBe('invokeai_img_blend');
    expect(harness.notices()).toHaveLength(0);
  });
});

// Task 39, finding 1: a regional reference image is only usable once an image is
// assigned (the settings UI now wires drop/upload → config.image). This guards
// the resolver seam: refs without an image are dropped; refs with an image + a
// compatible model survive into the graph inputs.
describe('resolveRegionalReferenceImages', () => {
  const asset = { imageName: 'ref.png' } as GenerateReferenceImageAsset;
  const ipAdapterModel = { base: 'sd-1', key: 'ipa', name: 'IP Adapter', type: 'ip_adapter' };
  const fluxReduxModel = { base: 'flux', key: 'redux', name: 'FLUX Redux', type: 'flux_redux' };

  const ipAdapterRef = (image: GenerateReferenceImageAsset | null): GenerateReferenceImage => ({
    config: {
      beginEndStepPct: [0, 1],
      clipVisionModel: 'ViT-H',
      image,
      method: 'full',
      model: ipAdapterModel,
      type: 'ip_adapter',
      weight: 1,
    },
    id: 'ref-ipa',
    isEnabled: true,
  });

  const fluxReduxRef = (image: GenerateReferenceImageAsset | null): GenerateReferenceImage => ({
    config: { image, imageInfluence: 'highest', model: fluxReduxModel, type: 'flux_redux' },
    id: 'ref-redux',
    isEnabled: true,
  });

  it('drops an IP-Adapter ref with no image assigned', () => {
    expect(resolveRegionalReferenceImages({ referenceImages: [ipAdapterRef(null)] }, 'sd-1')).toEqual([]);
  });

  it('keeps an IP-Adapter ref once an image is assigned (non-FLUX base)', () => {
    const inputs = resolveRegionalReferenceImages({ referenceImages: [ipAdapterRef(asset)] }, 'sd-1');
    expect(inputs).toHaveLength(1);
    expect(inputs[0]).toMatchObject({ id: 'ref-ipa', imageName: 'ref.png', type: 'ip_adapter' });
  });

  it('drops a FLUX Redux ref with no image assigned', () => {
    expect(resolveRegionalReferenceImages({ referenceImages: [fluxReduxRef(null)] }, 'flux')).toEqual([]);
  });

  it('keeps a FLUX Redux ref once an image is assigned (FLUX base)', () => {
    const inputs = resolveRegionalReferenceImages({ referenceImages: [fluxReduxRef(asset)] }, 'flux');
    expect(inputs).toHaveLength(1);
    expect(inputs[0]).toMatchObject({ id: 'ref-redux', imageName: 'ref.png', type: 'flux_redux' });
  });

  it('drops a disabled ref even when an image is assigned', () => {
    const disabled = { ...ipAdapterRef(asset), isEnabled: false };
    expect(resolveRegionalReferenceImages({ referenceImages: [disabled] }, 'sd-1')).toEqual([]);
  });
});

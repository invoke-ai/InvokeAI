import type { GenerateModelConfig, MainModelConfig } from '@features/generation/contracts';
import type {
  CanvasDocumentContractV3,
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasStateContractV3,
  RegionalGuidanceReferenceImage,
  RegionalGuidanceReferenceImageAsset,
} from '@workbench/canvas-engine/contracts';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { WorkbenchState } from '@workbench/projectContracts';
import type { WorkbenchAction } from '@workbench/workbenchState.testing';
import type { WorkbenchCommands } from '@workbench/workbenchStore';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { getDocumentLeaves } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import {
  composeForGeneration,
  createCompositeDedupeCache,
  type GenerationCompositeExecutorDeps,
  type GenerationCompositeHost,
} from '@workbench/canvas-operations/api';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { RunCanvasInvocationDeps } from './prepareCanvasInvocation';

import { DEFAULT_CANVAS_COMPOSITING } from './canvasCompositing';
import { DEFAULT_CANVAS_SCALING } from './canvasScaling';
import {
  prepareCanvasInvocation,
  resolveRegionalReferenceImages,
  runCanvasInvocation,
} from './prepareCanvasInvocation';

const canvasBoundaryMocks = vi.hoisted(() => ({
  getCanvasEngine: vi.fn(),
  getCanvasOperations: vi.fn(),
  generationDeviceOptions: [] as { device: string; name: string }[],
}));

vi.mock('@workbench/canvas-operations/api', async (importOriginal) => {
  const original = (await importOriginal()) as Record<string, unknown>;

  return {
    ...original,
    getCanvasEngine: canvasBoundaryMocks.getCanvasEngine,
    getCanvasOperations: canvasBoundaryMocks.getCanvasOperations,
  };
});

vi.mock('@features/queue/devices', async (importOriginal) => {
  const original = (await importOriginal()) as Record<string, unknown>;

  return {
    ...original,
    getGenerationDevicesSnapshot: () => ({
      error: null,
      loadState: 'loaded' as const,
      options: canvasBoundaryMocks.generationDeviceOptions,
      setting: 'auto' as const,
    }),
  };
});

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
  seedMode: 'fixed',
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

const makeDoc = (layers: CanvasRasterLayerContractV2[], size = 64): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: size, width: size, x: 0, y: 0 },
  height: size,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
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
  surfaceIds: string[];
  releaseRasterSnapshot: ReturnType<typeof vi.fn>;
  dedupe: ReturnType<typeof createCompositeDedupeCache>;
  executorDeps: GenerationCompositeExecutorDeps;
  /** The fake engine seam the REAL `composeForGeneration` operation runs against. */
  host: GenerationCompositeHost;
  submittedGraphs: () => Extract<WorkbenchAction, { type: 'submitCanvasInvocationSnapshot' }>[];
  notices: () => { message?: string }[];
}

interface HarnessOptions {
  document?: CanvasDocumentContractV3;
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
  outputOnlyMaskedRegions?: boolean;
  scaling?: RunCanvasInvocationDeps['scaling'];
  generateValues?: Record<string, unknown>;
  /** Replaces the uniform-alpha coverage readback. */
  readImageData?: GenerationCompositeExecutorDeps['readImageData'];
  controlModelMemory?: RunCanvasInvocationDeps['controlModelMemory'];
}

const makeHarness = (options: HarnessOptions = {}): Harness => {
  const stub = createTestStubRasterBackend();
  const events: string[] = [];
  const surfaceIds: string[] = [];
  const model = options.model ?? sd1Model;
  const alpha = options.alpha ?? 255;

  const backend = {
    createSurface: (w: number, h: number): RasterSurface => stub.createSurface(w, h),
    encodeSurface: (surface: RasterSurface): Promise<Blob> => stub.encodeSurface(surface),
  };

  const layerSurfaces = new Map<string, RasterSurface>();
  const getLayerSurface = (layerId: string): Promise<{ surface: RasterSurface; rect: Rect }> => {
    events.push('getLayerSurface');
    surfaceIds.push(layerId);
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
  const releaseRasterSnapshot = vi.fn();
  const dedupe = createCompositeDedupeCache();

  const makeCanvas = (document: CanvasDocumentContractV3): CanvasStateContractV3 => ({
    document: structuredClone(document),
    documentRevision: 0,
    snapshots: [],
    stagingArea: {
      areThumbnailsVisible: true,
      autoSwitchMode: 'off',
      isVisible: false,
      pendingImageIds: [],
      pendingImages: [],
      selectedImageIndex: 0,
    },
    version: 3,
  });

  const executorDeps: GenerationCompositeExecutorDeps = {
    backend,
    hashBlob: (blob: Blob) => blob.text(),
    readImageData: options.readImageData ?? ((_surface, rect) => uniformImageData(rect.width, rect.height, alpha)),
    uploadImage: uploadImage as (blob: Blob) => Promise<CanvasImageUploadResult>,
  };

  // Run real composeForGeneration against stub raster/upload/release dependencies and a harness-owned dedupe
  // cache.
  const host: GenerationCompositeHost = {
    captureDocumentSnapshot: () => ({
      canvas: makeCanvas(options.document ?? makeDoc([rasterLayer('layer-a')])),
      documentGeneration: 0,
    }),
    captureRasterSnapshot: async (documentSnapshot, layerIds) => {
      const layerSurfaces = new Map<string, { surface: RasterSurface; rect: Rect }>();
      for (const layerId of layerIds) {
        layerSurfaces.set(layerId, await getLayerSurface(layerId));
      }
      return {
        status: 'ok',
        snapshot: {
          canvas: documentSnapshot.canvas,
          documentGeneration: documentSnapshot.documentGeneration,
          emptyLayerIds: new Set<string>(),
          layerSurfaces,
          release: releaseRasterSnapshot,
        },
      };
    },
    dedupe,
    getCompositeExecutorDeps: () => executorDeps,
  };

  const deps: RunCanvasInvocationDeps = {
    composeForGeneration: (composeOptions) => composeForGeneration(host, composeOptions),
    controlModelMemory: options.controlModelMemory ?? {},
    compositing: {
      ...DEFAULT_CANVAS_COMPOSITING,
      outputOnlyMaskedRegions: options.outputOnlyMaskedRegions ?? true,
    },
    destination: options.destination ?? 'canvas',
    // The harness's tiny frames would grow under the default auto scaling; wiring tests read them at frame size.
    scaling: options.scaling ?? { ...DEFAULT_CANVAS_SCALING, method: 'none' },
    commands: {
      generation: {
        submitCanvas: (payload) => dispatch({ ...payload, type: 'submitCanvasInvocationSnapshot' }),
      } as WorkbenchCommands['generation'],
      notifications: {
        add: (notice) => dispatch({ ...notice, type: 'recordNotice' }),
      } as WorkbenchCommands['notifications'],
    },
    flushPendingUploads: flushPendingUploads as () => Promise<void>,
    // Production words rejections from the locale; the harness records the structured rejection it was handed.
    formatControlLayerError: ({ code, layerName }) => `${layerName}: ${code}`,
    generateValues: { ...generateValuesFor(model), ...options.generateValues },
    inFlight: options.inFlight ?? new Set<string>(),
    models: options.models,
    projectId: options.projectId ?? 'project-1',
    projectSettings: { useCpuNoise: true, useNsfwChecker: false, useWatermark: false },
    signal: new AbortController().signal,
    strength: options.strength ?? 0.75,
  };

  const submittedGraphs = () =>
    dispatch.mock.calls.map(([action]) => action).filter((action) => action.type === 'submitCanvasInvocationSnapshot');
  const notices = () =>
    dispatch.mock.calls.map(([action]) => action).filter((action) => action.type === 'recordNotice');

  return {
    dedupe,
    deps,
    dispatch,
    executorDeps,
    events,
    flushPendingUploads,
    host,
    notices,
    releaseRasterSnapshot,
    submittedGraphs,
    surfaceIds,
    uploadImage,
  };
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

seedArchitectureCapabilities();

describe('prepareCanvasInvocation generation-device boundary', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    canvasBoundaryMocks.getCanvasEngine.mockReset();
    canvasBoundaryMocks.getCanvasOperations.mockReset();
    canvasBoundaryMocks.generationDeviceOptions = [];
  });

  it('passes the XPU runtime snapshot into the compiled canvas graph metadata', async () => {
    const harness = makeHarness({ document: makeDoc([]) });
    canvasBoundaryMocks.generationDeviceOptions = [{ device: 'xpu:0', name: 'Intel Arc' }];
    canvasBoundaryMocks.getCanvasEngine.mockReturnValue({
      lifecycle: { flushPendingUploads: harness.flushPendingUploads },
    });
    canvasBoundaryMocks.getCanvasOperations.mockReturnValue({
      composeForGeneration: (options: Parameters<typeof composeForGeneration>[1]) =>
        composeForGeneration(harness.host, options),
    });

    await prepareCanvasInvocation({
      commands: harness.deps.commands,
      formatControlLayerError: harness.deps.formatControlLayerError,
      compositing: harness.deps.compositing,
      destination: harness.deps.destination,
      generateValues: harness.deps.generateValues,
      models: harness.deps.models,
      projectId: harness.deps.projectId,
      canvasValues: { scaleMethod: 'manual', scaledHeight: 768, scaledWidth: 1024 },
      projectSettings: { useCpuNoise: false, useNsfwChecker: false, useWatermark: false },
      strength: harness.deps.strength,
    });

    const nodes = harness.submittedGraphs()[0]?.graph.backendGraph?.nodes ?? {};
    const metadata = Object.values(nodes).find((node) => node.type === 'core_metadata');

    expect(metadata?.rand_device).toBe('xpu');
  });

  it("submits the caller's prompt expansion with the sample seed it drew", async () => {
    const harness = makeHarness({ document: makeDoc([]) });
    canvasBoundaryMocks.getCanvasEngine.mockReturnValue({
      lifecycle: { flushPendingUploads: harness.flushPendingUploads },
    });
    canvasBoundaryMocks.getCanvasOperations.mockReturnValue({
      composeForGeneration: (options: Parameters<typeof composeForGeneration>[1]) =>
        composeForGeneration(harness.host, options),
    });

    await prepareCanvasInvocation({
      commands: harness.deps.commands,
      destination: harness.deps.destination,
      expansion: { positivePrompts: ['a green cat'], positivePromptsSampleSeed: 7 },
      formatControlLayerError: harness.deps.formatControlLayerError,
      generateValues: harness.deps.generateValues,
      models: harness.deps.models,
      projectId: harness.deps.projectId,
      projectSettings: { useCpuNoise: false, useNsfwChecker: false, useWatermark: false },
      strength: harness.deps.strength,
    });

    expect(harness.submittedGraphs()).toEqual([
      expect.objectContaining({ positivePrompts: ['a green cat'], positivePromptsSampleSeed: 7 }),
    ]);
  });
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
    expect(harness.dedupe.byKey.size).toBeGreaterThan(0);
  });

  it('denoises at the persisted manual processing size and restores the bbox', async () => {
    const harness = makeHarness({ scaling: { height: 768, method: 'manual', width: 1024 } });

    await runCanvasInvocation(harness.deps);

    const nodes = harness.submittedGraphs()[0]!.graph.backendGraph!.nodes;
    expect(nodes.noise).toMatchObject({ height: 768, width: 1024 });
    expect(nodes.canvas_resize_initial_to_processing).toMatchObject({ height: 768, width: 1024 });
    expect(nodes.canvas_output).toMatchObject({ type: 'img_resize' });
  });

  it('reuses committed composite uploads on a later successful invocation', async () => {
    const harness = makeHarness();

    await runCanvasInvocation(harness.deps);
    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(2);
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
  });

  it('dispatches resolved prompt and seed metadata with the compiled canvas graph', async () => {
    const harness = makeHarness();
    harness.deps.generateValues = {
      ...harness.deps.generateValues,
      negativePrompt: 'low quality',
      positivePrompt: 'a resolved canvas prompt',
      seed: 123,
      seedMode: 'fixed',
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
        seedMode: 'fixed',
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

  it('carries Invoke Front to the queued Canvas snapshot only when asked', async () => {
    const appended = makeHarness({ document: makeDoc([]) });
    const front = makeHarness({ document: makeDoc([]) });
    front.deps.prepend = true;

    await runCanvasInvocation(appended.deps);
    await runCanvasInvocation(front.deps);

    expect(appended.submittedGraphs()[0]).not.toHaveProperty('prepend');
    expect(front.submittedGraphs()[0]?.prepend).toBe(true);
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

  it('aborts with a visible notice when pending paint persistence fails', async () => {
    const harness = makeHarness({
      flushPendingUploads: () => Promise.reject(new Error('Canvas pixel persistence failed for 1 layer.')),
    });

    await runCanvasInvocation(harness.deps);

    expect(harness.notices()).toEqual([
      expect.objectContaining({ message: 'Canvas pixel persistence failed for 1 layer.' }),
    ]);
    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.uploadImage).not.toHaveBeenCalled();
  });

  it('plans from the POST-flush document (re-reads getDocument after the flush barrier)', async () => {
    // Reread the document after upload flush updates paint sources; pre-flush references would corrupt dedupe and
    // omit newly painted content.
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
    harness.host.captureDocumentSnapshot = () => ({
      canvas: {
        document: structuredClone(flushed ? postDoc : preDoc),
        documentRevision: 0,
        snapshots: [],
        stagingArea: {
          areThumbnailsVisible: true,
          autoSwitchMode: 'off',
          isVisible: false,
          pendingImageIds: [],
          pendingImages: [],
          selectedImageIndex: 0,
        },
        version: 3,
      },
      documentGeneration: 0,
    });

    await runCanvasInvocation(harness.deps);

    // The composite rasterized the post-flush layer, never the stale pre-flush one.
    expect(harness.surfaceIds).toContain('post');
    expect(harness.surfaceIds).not.toContain('pre');
  });

  it('does not submit or publish dedupe entries when the document becomes stale during raster detachment', async () => {
    const harness = makeHarness();
    const snapshot = {
      canvas: {
        document: makeDoc([rasterLayer('layer-a')]),
        documentRevision: 0,
        snapshots: [],
        stagingArea: {
          areThumbnailsVisible: true,
          autoSwitchMode: 'off' as const,
          isVisible: false,
          pendingImageIds: [],
          pendingImages: [],
          selectedImageIndex: 0,
        },
        version: 2 as const,
      },
      documentGeneration: 0,
    };
    Object.assign(harness.host, {
      captureDocumentSnapshot: vi.fn(() => snapshot),
      captureRasterSnapshot: vi.fn(() => Promise.resolve({ status: 'stale' as const })),
    });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices()).toHaveLength(1);
    expect(harness.dedupe.byKey).toHaveLength(0);
    expect(harness.dedupe.byHash).toHaveLength(0);
  });

  it.each(['aborted', 'not-ready', 'over-budget'] as const)(
    'does not submit when raster capture returns the typed %s outcome',
    async (status) => {
      const harness = makeHarness();
      harness.host.captureRasterSnapshot = vi.fn(() => Promise.resolve({ status }));

      await runCanvasInvocation(harness.deps);

      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.notices()).toHaveLength(1);
      expect(harness.dedupe.byKey).toHaveLength(0);
      expect(harness.dedupe.byHash).toHaveLength(0);
    }
  );

  it('passes the orchestration AbortSignal into raster snapshot capture', async () => {
    const harness = makeHarness();
    const controller = new AbortController();
    const captureRasterSnapshot = vi.fn(() => Promise.resolve({ status: 'aborted' as const }));
    harness.host.captureRasterSnapshot = captureRasterSnapshot;
    harness.deps.signal = controller.signal;

    await runCanvasInvocation(harness.deps);

    expect(captureRasterSnapshot).toHaveBeenCalledWith(
      expect.any(Object),
      ['layer-a'],
      expect.objectContaining({ signal: controller.signal })
    );
    expect(harness.submittedGraphs()).toHaveLength(0);
  });

  it('does not start raster capture when the orchestration signal is already aborted', async () => {
    const harness = makeHarness();
    const controller = new AbortController();
    const captureRasterSnapshot = vi.fn(harness.host.captureRasterSnapshot);
    harness.host.captureRasterSnapshot = captureRasterSnapshot;
    harness.deps.signal = controller.signal;
    controller.abort(new DOMException('invoke cancelled', 'AbortError'));

    await runCanvasInvocation(harness.deps);

    expect(captureRasterSnapshot).not.toHaveBeenCalled();
    expect(harness.uploadImage).not.toHaveBeenCalled();
    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices()).toHaveLength(0);
  });

  it('quietly stops when its account signal expires during composition', async () => {
    const harness = makeHarness();
    const controller = new AbortController();
    harness.deps.signal = controller.signal;
    harness.deps.composeForGeneration = () => {
      controller.abort(new DOMException('account changed', 'AbortError'));
      return Promise.reject(controller.signal.reason);
    };

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices()).toHaveLength(0);
  });

  it('names the layer or group whose Clarity, Texture or Dehaze refused generation, and dispatches nothing', async () => {
    const harness = makeHarness();
    harness.deps.composeForGeneration = () =>
      Promise.resolve({ ownerId: 'g', ownerName: 'Sky', reason: 'too-large', status: 'adjustment-refused' });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices().map((notice) => notice.message)).toEqual([
      'Clarity, Texture and Dehaze can’t run on “Sky”: its content is larger than 16.8 megapixels (4096 × 4096 px). Disable or remove them, or crop the content, then invoke again.',
    ]);
  });

  it.each([
    [
      { layerId: 'layer-7', layerName: 'Sky', status: 'unreadable' } as const,
      '“Sky” could not be read (its image may have been deleted or damaged). Disable or delete that layer, then invoke again.',
    ],
    [{ status: 'not-ready' } as const, 'Some layers are still loading. Wait a moment, then invoke again.'],
  ])('explains a %o capture and dispatches nothing', async (result, message) => {
    const harness = makeHarness();
    harness.deps.composeForGeneration = () => Promise.resolve(result);

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices().map((notice) => notice.message)).toEqual([message]);
  });

  it('uses the frozen canvas and detached pixels after edits at composite, upload, and queue-dispatch boundaries', async () => {
    let liveDocument = makeDoc([rasterLayer('frozen-layer')], 96);
    liveDocument.bbox = { height: 64, width: 64, x: 7, y: 11 };
    const dispatch = vi.fn((_action: WorkbenchAction) => {
      liveDocument = { ...liveDocument, bbox: { height: 96, width: 96, x: 303, y: 404 } };
    });
    const harness = makeHarness({ dispatch, document: liveDocument });
    harness.host.captureDocumentSnapshot = () => ({
      canvas: {
        document: structuredClone(liveDocument),
        documentRevision: 4,
        snapshots: [],
        stagingArea: {
          areThumbnailsVisible: true,
          autoSwitchMode: 'off',
          isVisible: false,
          pendingImageIds: [],
          pendingImages: [],
          selectedImageIndex: 0,
        },
        version: 3,
      },
      documentGeneration: 9,
    });
    const captureRasterSnapshot = harness.host.captureRasterSnapshot;
    harness.host.captureRasterSnapshot = async (snapshot, layerIds, captureOptions) => {
      const result = await captureRasterSnapshot(snapshot, layerIds, captureOptions);
      // The composite inputs are fully detached at this boundary. Subsequent
      // edits must not alter any composite/upload/compile/dispatch input.
      liveDocument = {
        ...liveDocument,
        bbox: { height: 64, width: 64, x: 101, y: 202 },
        stacks: stacksFrom([rasterLayer('replacement-layer')]),
      };
      return result;
    };

    await runCanvasInvocation(harness.deps);

    expect(harness.surfaceIds).toEqual(['frozen-layer']);
    expect(harness.notices()).toEqual([]);
    expect(harness.submittedGraphs()).toHaveLength(1);
    expect(harness.submittedGraphs()[0]?.canvas.document.bbox).toEqual({ height: 64, width: 64, x: 7, y: 11 });
    expect(getDocumentLeaves(harness.submittedGraphs()[0]?.canvas.document)[0]?.id).toBe('frozen-layer');
    expect(harness.releaseRasterSnapshot).toHaveBeenCalledTimes(1);
    expect(harness.dedupe.byKey.size).toBeGreaterThan(0);
  });

  it.each(['coverage-scan', 'encode', 'upload'] as const)(
    'keeps the frozen transaction when the live bbox changes at the %s boundary',
    async (boundary) => {
      let liveDocument = makeDoc([rasterLayer('layer-a')]);
      const harness = makeHarness({ document: liveDocument });
      harness.host.captureDocumentSnapshot = () => ({
        canvas: {
          document: structuredClone(liveDocument),
          documentRevision: 2,
          snapshots: [],
          stagingArea: {
            areThumbnailsVisible: true,
            autoSwitchMode: 'off',
            isVisible: false,
            pendingImageIds: [],
            pendingImages: [],
            selectedImageIndex: 0,
          },
          version: 3,
        },
        documentGeneration: 3,
      });
      const editLiveBbox = (): void => {
        liveDocument = { ...liveDocument, bbox: { height: 64, width: 64, x: 88, y: 99 } };
      };
      if (boundary === 'coverage-scan') {
        const readImageData = harness.executorDeps.readImageData!;
        harness.executorDeps.readImageData = (...args) => {
          editLiveBbox();
          return readImageData(...args);
        };
      } else if (boundary === 'encode') {
        const encodeSurface = harness.executorDeps.backend.encodeSurface;
        harness.executorDeps.backend.encodeSurface = (...args) => {
          editLiveBbox();
          return encodeSurface(...args);
        };
      } else {
        const uploadImage = harness.executorDeps.uploadImage;
        harness.executorDeps.uploadImage = (...args) => {
          editLiveBbox();
          return uploadImage(...args);
        };
      }

      await runCanvasInvocation(harness.deps);

      expect(harness.submittedGraphs()).toHaveLength(1);
      expect(harness.submittedGraphs()[0]?.canvas.document.bbox).toEqual({ height: 64, width: 64, x: 0, y: 0 });
      expect(harness.releaseRasterSnapshot).toHaveBeenCalledTimes(1);
    }
  );

  it('records a notice and dispatches no snapshot when the graph fails validation', async () => {
    // The raster snapshot makes this an image-to-image run, and the record lists text-to-image only, so the notice
    // names the frame's content as the cause.
    const harness = makeHarness({ model: externalModel });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    const notices = harness.notices();
    expect(notices).toHaveLength(1);
    expect(notices[0]?.message).toBe(
      'OpenAI Image cannot use image content in the generation frame. Hide or move raster content out of the frame to generate.'
    );
    expect(harness.dedupe.byKey).toHaveLength(0);
    expect(harness.dedupe.byHash).toHaveLength(0);
    expect(harness.releaseRasterSnapshot).toHaveBeenCalledTimes(1);
  });

  it('generates from text for a text-only provider when the frame holds no visible pixels', async () => {
    // The raster layer's bounds overlap the frame, but every composited pixel there is transparent.
    const harness = makeHarness({ alpha: 0, model: externalModel });

    await runCanvasInvocation(harness.deps);

    expect(harness.notices()).toEqual([]);
    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.graph.backendGraph?.nodes.canvas_processing_output).toMatchObject({ mode: 'txt2img' });
    expect(submitted[0]?.graph.backendGraph?.nodes.canvas_processing_output).not.toHaveProperty('init_image');
  });

  it('generates from text for an editing provider when the frame holds no visible pixels', async () => {
    // Without this, the partial bounds overlap would read as outpaint, which hosted providers refuse.
    const model: GenerateModelConfig = { ...externalModel, capabilities: { modes: ['txt2img', 'img2img'] } };
    const harness = makeHarness({ alpha: 0, model });

    await runCanvasInvocation(harness.deps);

    expect(harness.notices()).toEqual([]);
    expect(harness.submittedGraphs()[0]?.graph.backendGraph?.nodes.canvas_processing_output).toMatchObject({
      mode: 'txt2img',
    });
  });

  it('refuses a text-only provider when any pixel in the frame is visible, even partly', async () => {
    const harness = makeHarness({ alpha: 1, model: externalModel });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices().map((notice) => notice.message)).toEqual([
      'OpenAI Image cannot use image content in the generation frame. Hide or move raster content out of the frame to generate.',
    ]);
  });

  it.each(['control', 'regional'] as const)(
    'rejects contributing external %s layers by name before enqueue',
    async (kind) => {
      const layer = kind === 'control' ? controlLayer('Canny guide') : regionalLayer('Hat region');
      const harness = makeHarness({ model: externalModel, document: docWithLayers([layer]) });
      await runCanvasInvocation(harness.deps);
      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.notices().map((notice) => notice.message)).toEqual([
        kind === 'control'
          ? 'OpenAI Image does not support control layer "Canny guide". Disable it to generate.'
          : 'OpenAI Image does not support regional guidance layer "Hat region". Disable it to generate.',
      ]);
      expect(harness.dedupe.byKey.size).toBe(0);
      expect(harness.releaseRasterSnapshot).toHaveBeenCalledTimes(1);
    }
  );

  it.each(['control', 'regional'] as const)('ignores disabled external %s layers', async (kind) => {
    const layer = kind === 'control' ? controlLayer('Canny guide') : regionalLayer('Hat region');
    const harness = makeHarness({ model: externalModel, document: docWithLayers([{ ...layer, isEnabled: false }]) });
    await runCanvasInvocation(harness.deps);
    expect(harness.notices()).toEqual([]);
    expect(harness.submittedGraphs()).toHaveLength(1);
  });

  it('submits an img2img-only provider through the real preparation path', async () => {
    const model: GenerateModelConfig = { ...externalModel, capabilities: { modes: ['img2img'] } };
    const harness = makeHarness({ model });
    await runCanvasInvocation(harness.deps);
    expect(harness.notices()).toEqual([]);
    expect(harness.submittedGraphs()).toHaveLength(1);
  });

  it('blocks advertised provider inpainting before enqueue', async () => {
    const model: GenerateModelConfig = { ...externalModel, capabilities: { modes: ['inpaint', 'img2img'] } };
    const harness = makeHarness({ model, document: docWithLayers([rasterLayer('base'), inpaintMaskLayer('mask')]) });
    await runCanvasInvocation(harness.deps);
    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices().map((notice) => notice.message)).toEqual([
      'OpenAI Image does not support inpainting on the canvas yet. Remove or disable the inpaint mask.',
    ]);
  });

  it('blocks oversized provider requests before enqueue', async () => {
    const model: GenerateModelConfig = {
      ...externalModel,
      capabilities: { modes: ['img2img'], max_image_size: { width: 32, height: 64 } },
    };
    const harness = makeHarness({ model });
    await runCanvasInvocation(harness.deps);
    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices().map((notice) => notice.message)).toEqual([
      'OpenAI Image supports a maximum size of 32×64.',
    ]);
  });

  it('counts the captured init image against the Seedream budget before enqueue', async () => {
    const model: GenerateModelConfig = {
      ...externalModel,
      name: 'Seedream',
      provider_id: 'seedream',
      provider_model_id: 'seedream-4-5-251128',
      capabilities: { modes: ['txt2img', 'img2img'], max_reference_images: 14, supports_reference_images: true },
    };
    const harness = makeHarness({
      model,
      generateValues: {
        ...generateValuesFor(model),
        referenceImages: Array.from({ length: 14 }, (_, i) => ({
          id: `reference-${i}`,
          isEnabled: true,
          config: {
            type: 'external_reference_image',
            image: { imageName: `reference-${i}.png`, width: 64, height: 64 },
          },
        })),
      },
    });
    await runCanvasInvocation(harness.deps);
    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices().map((notice) => notice.message)).toEqual([
      'Seedream supports at most 15 images total (reference + init + output), got 16.',
    ]);
  });

  it('discards provisional dedupe entries when synchronous queue dispatch fails', async () => {
    const dispatch = vi.fn((action: WorkbenchAction) => {
      if (action.type === 'submitCanvasInvocationSnapshot') {
        throw new Error('queue dispatch failed');
      }
    });
    const harness = makeHarness({ dispatch });

    await runCanvasInvocation(harness.deps);

    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
    expect(harness.dedupe.byKey).toHaveLength(0);
    expect(harness.dedupe.byHash).toHaveLength(0);
    expect(harness.notices()).toEqual([
      expect.objectContaining({ message: 'queue dispatch failed', type: 'recordNotice' }),
    ]);
  });

  it('records a notice and dispatches no snapshot when the composite upload fails', async () => {
    const uploadImage = vi.fn(() => Promise.reject(new Error('upload exploded')));
    const harness = makeHarness({ uploadImage });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    const notices = harness.notices();
    expect(notices).toHaveLength(1);
    expect(notices[0]?.message).toBe('upload exploded');
    expect(harness.dedupe.byKey).toHaveLength(0);
    expect(harness.dedupe.byHash).toHaveLength(0);
    expect(harness.releaseRasterSnapshot).toHaveBeenCalledTimes(1);
  });

  it('records the raster memory notice and dispatches nothing when the composite is over budget', async () => {
    const harness = makeHarness();
    harness.executorDeps.reserve = (bytes) => ({ availableBytes: 0, requestedBytes: bytes, status: 'over-budget' });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices().map((notice) => notice.message)).toEqual([
      'The canvas composite exceeds the available raster memory budget.',
    ]);
    expect(harness.releaseRasterSnapshot).toHaveBeenCalledTimes(1);
  });

  it('blocks a content-bearing control layer while the model list is still loading', async () => {
    const harness = makeHarness({ document: docWithLayers([controlLayer('control')]) });
    await runCanvasInvocation(harness.deps);
    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices().map((notice) => notice.message)).toEqual(['control: models_unavailable']);
  });

  it('blocks an untyped control layer whose model is no longer installed', async () => {
    const harness = makeHarness({
      document: docWithLayers([controlLayer('control', { model: 'uninstalled' })]),
      models: [],
    });
    await runCanvasInvocation(harness.deps);
    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices().map((notice) => notice.message)).toEqual(['control: missing_model']);
  });

  it('leaves out a nonempty control layer with neither a type nor a model, without blocking', async () => {
    const harness = makeHarness({ document: docWithLayers([controlLayer('control')]), models: [] });
    await runCanvasInvocation(harness.deps);
    expect(harness.notices()).toHaveLength(0);
    expect(harness.submittedGraphs()).toHaveLength(1);
    expect(harness.submittedGraphs()[0]?.graph.backendGraph?.nodes).not.toHaveProperty('control_net_control');
  });

  describe('typed control layers', () => {
    const controlModel = (key: string, base: string, type = 'controlnet', name = key) => ({
      base,
      file_size: 1,
      format: 'checkpoint',
      hash: `${key}-hash`,
      key,
      name,
      path: key,
      source: key,
      source_type: 'path' as const,
      type,
    });
    const typed = (
      id: string,
      controlType: NonNullable<CanvasControlLayerContract['controlType']>,
      detection: CanvasControlLayerContract['detection'] = null
    ): CanvasControlLayerContract => ({ ...controlLayer(id), controlType, detection });
    const fluxMain = { ...sd1Model, base: 'flux' as const, key: 'flux', name: 'FLUX' };

    it('runs the only model made for the type, as the kind that model runs', async () => {
      const harness = makeHarness({
        document: docWithLayers([typed('depth-layer', 'depth')]),
        models: [controlModel('sd1-depth', 'sd-1', 'controlnet', 'control_v11f1p_sd15_depth')],
      });

      await runCanvasInvocation(harness.deps);

      expect(harness.notices()).toHaveLength(0);
      expect(harness.submittedGraphs()[0]?.graph.backendGraph?.nodes['control_net_depth-layer']).toMatchObject({
        control_model: { key: 'sd1-depth' },
        type: 'controlnet',
      });
    });

    it('runs the remembered pick when several models fit, and asks without one', async () => {
      const models = [
        controlModel('depth-a', 'sd-1', 'controlnet', 'depth a'),
        controlModel('depth-b', 'sd-1', 'controlnet', 'depth b'),
      ];
      const asking = makeHarness({ document: docWithLayers([typed('depth-layer', 'depth')]), models });
      await runCanvasInvocation(asking.deps);
      expect(asking.submittedGraphs()).toHaveLength(0);
      expect(asking.notices()[0]?.message).toBe('depth-layer: choose_model');

      const remembered = makeHarness({
        controlModelMemory: { 'sd-1:depth': 'depth-b' },
        document: docWithLayers([typed('depth-layer', 'depth')]),
        models,
      });
      await runCanvasInvocation(remembered.deps);
      expect(remembered.notices()).toHaveLength(0);
      expect(
        remembered.submittedGraphs()[0]?.graph.backendGraph?.nodes['control_net_depth-layer']?.control_model
      ).toMatchObject({ key: 'depth-b' });
    });

    it('gives a FLUX union model the InstantX mode of the layer type', async () => {
      const components = {
        clipEmbedModel: { base: 'any', key: 'clip', name: 'CLIP Embed', type: 'clip_embed', variant: 'large' },
        t5EncoderModel: { base: 'any', key: 't5', name: 'T5 Encoder', type: 't5_encoder' },
        vae: { base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' },
      };
      const harness = makeHarness({
        document: docWithLayers([typed('pose-layer', 'pose')]),
        generateValues: components,
        model: fluxMain,
        models: [
          controlModel('flux-union', 'flux', 'controlnet', 'FLUX.1-dev-Controlnet-Union-Pro'),
          ...(Object.values(components) as unknown as NonNullable<RunCanvasInvocationDeps['models']>),
        ],
      });

      await runCanvasInvocation(harness.deps);

      expect(harness.notices()).toEqual([]);
      expect(harness.submittedGraphs()[0]?.graph.backendGraph?.nodes['control_net_pose-layer']).toMatchObject({
        instantx_control_mode: 4,
        type: 'flux_controlnet',
      });
    });

    it('blocks a layer whose type has no installed model', async () => {
      const harness = makeHarness({
        document: docWithLayers([typed('pose-layer', 'pose')]),
        models: [controlModel('sd1-depth', 'sd-1', 'controlnet', 'sd15 depth')],
      });
      await runCanvasInvocation(harness.deps);
      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.notices()[0]?.message).toBe('pose-layer: no_model_for_type');
    });

    it('blocks a layer whose map detection has not caught up', async () => {
      const harness = makeHarness({
        document: docWithLayers([
          typed('depth-layer', 'depth', { settings: {}, type: 'depth_anything_depth_estimation' }),
        ]),
        models: [controlModel('sd1-depth', 'sd-1', 'controlnet', 'sd15 depth')],
      });
      await runCanvasInvocation(harness.deps);
      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.notices()[0]?.message).toBe('depth-layer: detection_pending');
    });
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
    expect(harness.notices()[0]?.message).toBe('control: invalid_adapter_values');
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
    expect(harness.surfaceIds).toEqual([]);
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
    expect(harness.notices()[0]?.message).toBe('second: control_lora_limit');
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
    expect(harness.surfaceIds).toEqual(['z-layer']);
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
    expect(harness.notices()[0]?.message).toBe('second: z_image_control_limit');
  });

  describe('Qwen Image ControlNet', () => {
    const qwenImage = {
      ...sd1Model,
      base: 'qwen-image' as const,
      format: 'diffusers' as const,
      key: 'qwen-image',
      name: 'Qwen Image',
    };
    const union = {
      base: 'qwen-image',
      file_size: 1,
      format: 'checkpoint',
      hash: 'qwen-union-hash',
      key: 'qwen-union',
      name: 'Qwen Image ControlNet Union',
      path: 'qwen-union',
      source: 'qwen-union',
      source_type: 'path' as const,
      type: 'controlnet',
    };
    const qwenLayer = (id: string, values: Partial<CanvasControlLayerContract['adapter']> = {}) =>
      controlLayer(id, { controlMode: null, kind: 'qwen_image_control', model: union.key, weight: 0.85, ...values });

    it('prepares the control with the adapter values and the installed model', async () => {
      const harness = makeHarness({
        document: docWithLayers([qwenLayer('q-layer', { beginEndStepPct: [0.1, 0.9], weight: 0.7 })]),
        model: qwenImage,
        models: [union],
      });

      await runCanvasInvocation(harness.deps);

      expect(harness.notices()).toHaveLength(0);
      expect(harness.submittedGraphs()[0]?.graph.backendGraph?.nodes['qwen_image_control_q-layer']).toMatchObject({
        begin_step_percent: 0.1,
        control_context_scale: 0.7,
        control_model: {
          base: 'qwen-image',
          hash: 'qwen-union-hash',
          key: 'qwen-union',
          name: 'Qwen Image ControlNet Union',
          type: 'controlnet',
        },
        end_step_percent: 0.9,
        type: 'qwen_image_control',
      });
      expect(harness.surfaceIds).toEqual(['q-layer']);
    });

    it('hands the control the uploads the inpaint graph itself reads when the canvas inpaints', async () => {
      const harness = makeHarness({
        document: docWithLayers([rasterLayer('base'), qwenLayer('q-layer'), inpaintMaskLayer('mask')]),
        model: qwenImage,
        models: [union],
      });

      await runCanvasInvocation(harness.deps);

      const submitted = harness.submittedGraphs();
      expect(submitted[0]?.graph.label).toBe('Qwen Image inpaint');
      const nodes = submitted[0]!.graph.backendGraph!.nodes;
      expect(nodes['qwen_image_control_q-layer']?.inpaint_image).toEqual(nodes.create_gradient_mask?.image);
      expect(nodes['qwen_image_control_mask_q-layer']).toMatchObject({
        image: nodes.create_gradient_mask?.mask,
        max: 0,
        min: 255,
        type: 'img_lerp',
      });
    });

    it('blocks a second enabled nonempty control', async () => {
      const harness = makeHarness({
        document: docWithLayers([qwenLayer('first'), qwenLayer('second')]),
        model: qwenImage,
        models: [union],
      });

      await runCanvasInvocation(harness.deps);

      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.notices()[0]?.message).toBe('second: qwen_image_control_limit');
    });

    it('blocks the control on a Qwen Image Edit model', async () => {
      const harness = makeHarness({
        document: docWithLayers([qwenLayer('layer')]),
        model: { ...qwenImage, variant: 'edit' },
        models: [union],
      });

      await runCanvasInvocation(harness.deps);

      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.notices()[0]?.message).toBe('layer: qwen_image_edit_control');
    });
  });

  describe('FLUX.2 [dev] ControlNet', () => {
    const flux2Dev = {
      ...sd1Model,
      base: 'flux2' as const,
      format: 'diffusers' as const,
      key: 'flux2-dev',
      name: 'FLUX.2 [dev]',
      variant: 'dev',
    };
    const union = {
      base: 'flux2',
      file_size: 1,
      format: 'checkpoint',
      hash: 'flux2-union-hash',
      key: 'flux2-union',
      name: 'FLUX.2 [dev] ControlNet Union',
      path: 'flux2-union',
      source: 'flux2-union',
      source_type: 'path' as const,
      type: 'controlnet',
    };
    const flux2Layer = (id: string, values: Partial<CanvasControlLayerContract['adapter']> = {}) =>
      controlLayer(id, { controlMode: null, kind: 'flux2_control', model: union.key, weight: 0.75, ...values });

    it('prepares the control with the adapter values and the installed model', async () => {
      const harness = makeHarness({
        document: docWithLayers([flux2Layer('f-layer', { beginEndStepPct: [0.1, 0.9], weight: 0.65 })]),
        model: flux2Dev,
        models: [union],
      });

      await runCanvasInvocation(harness.deps);

      expect(harness.notices()).toHaveLength(0);
      expect(harness.submittedGraphs()[0]?.graph.backendGraph?.nodes['flux2_control_f-layer']).toMatchObject({
        begin_step_percent: 0.1,
        control_context_scale: 0.65,
        control_model: { base: 'flux2', hash: 'flux2-union-hash', key: 'flux2-union', type: 'controlnet' },
        end_step_percent: 0.9,
        type: 'flux2_control',
      });
      expect(harness.surfaceIds).toEqual(['f-layer']);
    });

    it('hands the control the uploads the inpaint graph itself reads when the canvas inpaints', async () => {
      const harness = makeHarness({
        document: docWithLayers([rasterLayer('base'), flux2Layer('f-layer'), inpaintMaskLayer('mask')]),
        model: flux2Dev,
        models: [union],
      });

      await runCanvasInvocation(harness.deps);

      const nodes = harness.submittedGraphs()[0]!.graph.backendGraph!.nodes;
      expect(nodes['flux2_control_f-layer']?.inpaint_image).toEqual(nodes.create_gradient_mask?.image);
      expect(nodes['flux2_control_mask_f-layer']).toMatchObject({
        image: nodes.create_gradient_mask?.mask,
        max: 0,
        min: 255,
        type: 'img_lerp',
      });
    });

    it('blocks a second enabled nonempty control', async () => {
      const harness = makeHarness({
        document: docWithLayers([flux2Layer('first'), flux2Layer('second')]),
        model: flux2Dev,
        models: [union],
      });

      await runCanvasInvocation(harness.deps);

      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.notices()[0]?.message).toBe('second: flux2_control_limit');
    });

    it('blocks the control on a FLUX.2 Klein model', async () => {
      const harness = makeHarness({
        document: docWithLayers([flux2Layer('layer')]),
        model: { ...flux2Dev, key: 'flux2-klein', name: 'FLUX.2 Klein 9B', variant: 'klein_9b' },
        models: [union],
      });

      await runCanvasInvocation(harness.deps);

      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.notices()[0]?.message).toBe('layer: flux2_klein_control');
    });
  });

  describe('Anima ControlNet-LLLite', () => {
    const anima = { ...sd1Model, base: 'anima' as const, key: 'anima', name: 'Anima' };
    const installed = (key: string, base: string, type: string, extra: Record<string, unknown> = {}) => ({
      base,
      file_size: 1,
      format: 'checkpoint',
      hash: `${key}-hash`,
      key,
      name: key,
      path: key,
      source: key,
      source_type: 'path' as const,
      type,
      ...extra,
    });
    const components = [
      installed('qwen3', 'any', 'qwen3_encoder', { variant: 'qwen3_06b' }),
      installed('anima-vae', 'anima', 'vae'),
    ];
    const sketch = installed('sketch', 'anima', 'controlnet', { cond_in_channels: 3 });
    const depth = installed('depth', 'anima', 'controlnet', { cond_in_channels: 3 });
    const inpainting = installed('inpainting', 'anima', 'controlnet', { cond_in_channels: 4 });
    const lllite = (id: string, model: string, weight = 1) =>
      controlLayer(id, { beginEndStepPct: [0.1, 0.9], controlMode: null, kind: 'anima_lllite', model, weight });
    const run = async (layers: CanvasLayerContract[]) => {
      const harness = makeHarness({
        document: docWithLayers(layers),
        // Generate keeps component selections; the canvas does not pick them from the catalog.
        generateValues: { qwen3EncoderModel: components[0], vae: components[1] },
        model: anima,
        models: [...components, sketch, depth, inpainting],
      });
      await runCanvasInvocation(harness.deps);
      return harness;
    };

    it('submits each adapter with its resolved model through the control_lllite collector', async () => {
      const harness = await run([lllite('a', sketch.key, 0.6), lllite('b', depth.key)]);

      expect(harness.notices()).toEqual([]);
      const graph = harness.submittedGraphs()[0]?.graph.backendGraph;
      expect(graph?.nodes.anima_lllite_a).toMatchObject({
        begin_step_percent: 0.1,
        control_model: { base: 'anima', hash: 'sketch-hash', key: 'sketch', name: 'sketch', type: 'controlnet' },
        end_step_percent: 0.9,
        type: 'anima_lllite',
        weight: 0.6,
      });
      expect(graph?.nodes.anima_lllite_b).toMatchObject({ control_model: { key: 'depth' }, weight: 1 });
      // The identifier is the backend's model field; channel facts stay on the frontend.
      expect(graph?.nodes.anima_lllite_a?.control_model).not.toHaveProperty('cond_in_channels');
      expect(
        graph?.edges.filter((edge) => edge.destination.field === 'control_lllite').map((edge) => edge.source.node_id)
      ).toEqual(['control_lllite_collector']);
    });

    it.each([
      ['the same model twice', () => [lllite('a', sketch.key), lllite('b', sketch.key)], 'b: duplicate_lllite_model'],
      ['an inpainting adapter', () => [lllite('a', inpainting.key)], 'a: lllite_inpaint_adapter'],
      [
        'a layer saved as ControlNet before LLLite was supported',
        () => [controlLayer('old', { kind: 'controlnet', model: sketch.key })],
        'old: switch_adapter_kind',
      ],
    ])('blocks %s before submitting', async (_label, layers, message) => {
      const harness = await run(layers());

      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.notices()[0]?.message).toBe(message);
    });
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

// ---- Regional guidance through the real composite pipeline ---------------

const animaModel: MainModelConfig = { base: 'anima', key: 'anima-model', name: 'Anima', type: 'main' };
const animaGenerateValues = (): Record<string, unknown> => ({
  ...generateValuesFor(animaModel),
  cfgScale: 4,
  qwen3EncoderModel: { base: 'any', key: 'qwen3', name: 'Qwen3 Encoder', type: 'qwen3_encoder', variant: 'qwen3_06b' },
  vae: { base: 'anima', key: 'anima-vae', name: 'Anima VAE', type: 'vae' },
});

const regionalLayer = (
  id: string,
  overrides: Partial<
    Pick<
      CanvasLayerContract & { type: 'regional_guidance' },
      'positivePrompt' | 'negativePrompt' | 'autoNegative' | 'referenceImages' | 'layout'
    >
  > = {}
): CanvasLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: { height: 64, imageName: `${id}-bmp`, width: 64 }, fill: { color: '#00ff00', style: 'solid' } },
  name: id,
  negativePrompt: null,
  opacity: 0.5,
  positivePrompt: 'a red hat',
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
  ...overrides,
});

describe('runCanvasInvocation — regional guidance on a Qwen3 base', () => {
  const makeAnimaHarness = (layer: CanvasLayerContract) => {
    const harness = makeHarness({ document: docWithLayers([layer]), model: animaModel });
    harness.deps.generateValues = animaGenerateValues();
    return harness;
  };

  it('composites an Anima region and grafts its masked prompt into the positive collector', async () => {
    const harness = makeAnimaHarness(regionalLayer('rg'));

    await runCanvasInvocation(harness.deps);

    expect(harness.notices()).toHaveLength(0);
    // The only upload is the region's own mask composite (no raster content → no base composite).
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
    const graph = harness.submittedGraphs()[0]?.graph.backendGraph;
    expect(graph?.nodes.rg_pos_cond_rg).toMatchObject({ prompt: 'a red hat', type: 'anima_text_encoder' });
    expect(graph?.nodes.rg_mask_to_tensor_rg).toMatchObject({
      image: { image_name: 'composite-1.png' },
      type: 'alpha_mask_to_tensor',
    });
    expect(
      graph?.edges.some(
        (edge) => edge.source.node_id === 'rg_pos_cond_rg' && edge.destination.node_id === 'pos_cond_collect'
      )
    ).toBe(true);
  });

  it.each([
    ['a negative prompt', { negativePrompt: 'blurry' }],
    ['auto-negative', { autoNegative: true }],
    [
      'a complete IP-Adapter reference',
      {
        referenceImages: [
          {
            config: {
              beginEndStepPct: [0, 1],
              clipVisionModel: 'ViT-H',
              image: { imageName: 'ref.png' },
              method: 'full',
              model: { base: 'anima', key: 'ipa', name: 'IP Adapter', type: 'ip_adapter' },
              type: 'ip_adapter',
              weight: 1,
            },
            id: 'ref',
            isEnabled: true,
          },
        ],
      },
    ],
  ] as const)(
    'still submits the positive prompt of an Anima region carrying %s, without the extra',
    async (_label, overrides) => {
      const harness = makeAnimaHarness(regionalLayer('rg', overrides as Parameters<typeof regionalLayer>[1]));

      await runCanvasInvocation(harness.deps);

      expect(harness.notices()).toHaveLength(0);
      const graph = harness.submittedGraphs()[0]?.graph.backendGraph;
      expect(graph?.nodes.rg_pos_cond_rg).toMatchObject({ prompt: 'a red hat', type: 'anima_text_encoder' });
      expect(graph?.nodes.rg_neg_cond_rg).toBeUndefined();
      expect(graph?.nodes.rg_pos_cond_inverted_rg).toBeUndefined();
      expect(graph?.nodes.ip_adapter_ref).toBeUndefined();
      const negativeSources = graph?.edges
        .filter((edge) => edge.destination.node_id === 'neg_cond_collect')
        .map((edge) => edge.source.node_id);
      expect(negativeSources).toEqual(['neg_cond']);
    }
  );

  it('skips an Anima region that carries only a negative prompt', async () => {
    const harness = makeAnimaHarness(regionalLayer('rg', { negativePrompt: 'blurry', positivePrompt: null }));

    await runCanvasInvocation(harness.deps);

    expect(harness.notices()).toHaveLength(0);
    expect(harness.uploadImage).not.toHaveBeenCalled();
    const graph = harness.submittedGraphs()[0]?.graph.backendGraph;
    expect(graph?.nodes.denoise_latents?.type).toBe('anima_denoise');
    expect(graph?.nodes.rg_mask_to_tensor_rg).toBeUndefined();
  });
});

describe('runCanvasInvocation — layout regions on MOSS-Image', () => {
  const mossModel: MainModelConfig = {
    base: 'moss-image',
    format: 'diffusers',
    key: 'moss-model',
    name: 'MOSS-Image',
    type: 'main',
  };
  /** Every regional mask covers x 8–23, y 4–35 of the 64×64 bbox. */
  const maskScan: GenerationCompositeExecutorDeps['readImageData'] = (_surface, rect) => {
    const image = uniformImageData(rect.width, rect.height, 0);
    for (let y = 4; y < 36; y += 1) {
      for (let x = 8; x < 24; x += 1) {
        image.data[(y * rect.width + x) * 4 + 3] = 255;
      }
    }
    return image;
  };
  const makeMossHarness = (
    layers: CanvasLayerContract[],
    generateValues: Record<string, unknown> = {},
    readImageData = maskScan
  ) =>
    makeHarness({
      document: docWithLayers(layers),
      generateValues,
      model: mossModel,
      readImageData,
    });

  it("places each region's mask bounds and role in the layout prompt, skipping a text region without text", async () => {
    // Each composited mask is read back once; identical masks would share one upload, so count the readbacks.
    const readImageData = vi.fn(maskScan);
    const harness = makeMossHarness(
      [
        regionalLayer('sign', { layout: { kind: 'text', text: 'OPEN' }, positivePrompt: 'a neon sign' }),
        regionalLayer('blank', { layout: { kind: 'text' }, positivePrompt: 'a poster' }),
      ],
      {},
      readImageData
    );

    await runCanvasInvocation(harness.deps);

    expect(harness.notices()).toHaveLength(0);
    // The text region without text never composites.
    expect(readImageData).toHaveBeenCalledTimes(1);
    const graph = harness.submittedGraphs()[0]?.graph.backendGraph;
    expect(graph?.nodes.layout_prompt).toMatchObject({
      height: 64,
      regions: [{ bbox: [8, 4, 24, 36], caption: 'a neon sign', kind: 'text', text: 'OPEN' }],
      type: 'moss_image_layout_prompt',
      width: 64,
    });
    expect(Object.values(graph?.nodes ?? {}).some((node) => node.type === 'alpha_mask_to_tensor')).toBe(false);
  });

  it('blocks the invocation with a notice when regions would be laid out around a blank prompt', async () => {
    const harness = makeMossHarness([regionalLayer('hero', { layout: { kind: 'character' } })], {
      positivePrompt: '',
    });

    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(0);
    expect(harness.notices()).toEqual([
      expect.objectContaining({
        message:
          'MOSS-Image needs a prompt to place regional guidance in its layout. Enter a prompt or disable the regional guidance layers.',
      }),
    ]);
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

const docWithLayers = (layers: CanvasLayerContract[], size = 64): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: size, width: size, x: 0, y: 0 },
  height: size,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
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
    expect(nodes.canvas_output?.type).toBe('apply_mask_to_image');
    // The gradient mask consumes the uploaded grayscale mask (base upload is composite-1).
    expect((nodes.create_gradient_mask?.mask as { image_name?: string } | undefined)?.image_name).toBeDefined();
    expect(harness.notices()).toHaveLength(0);
    // The mask layer was rasterized for the grayscale composite.
    expect(harness.surfaceIds).toEqual(['base', 'mask']);
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
    expect(nodes.canvas_output?.type).toBe('apply_mask_to_image');
    expect(harness.notices()).toHaveLength(0);
  });

  it('composites the generated region over the source when masked-only output is disabled', async () => {
    const doc = docWithLayers([rasterLayer('base'), inpaintMaskLayer('mask')]);
    const harness = makeHarness({ document: doc, outputOnlyMaskedRegions: false });

    await runCanvasInvocation(harness.deps);

    const nodes = harness.submittedGraphs()[0]!.graph.backendGraph!.nodes;
    expect(nodes.canvas_output).toMatchObject({
      layer_base: expect.objectContaining({ image_name: expect.any(String) }),
      type: 'invokeai_img_blend',
    });
  });
});

// Regional references require an assigned image and compatible model before reaching graph inputs.
describe('resolveRegionalReferenceImages', () => {
  const asset = { imageName: 'ref.png' } as RegionalGuidanceReferenceImageAsset;
  const ipAdapterModel = { base: 'sd-1', key: 'ipa', name: 'IP Adapter', type: 'ip_adapter' };
  const fluxReduxModel = { base: 'flux', key: 'redux', name: 'FLUX Redux', type: 'flux_redux' };

  const ipAdapterRef = (image: RegionalGuidanceReferenceImageAsset | null): RegionalGuidanceReferenceImage => ({
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

  const fluxReduxRef = (image: RegionalGuidanceReferenceImageAsset | null): RegionalGuidanceReferenceImage => ({
    config: { image, imageInfluence: 'highest', model: fluxReduxModel, type: 'flux_redux' },
    id: 'ref-redux',
    isEnabled: true,
  });

  it('drops an IP-Adapter ref with no image assigned', () => {
    expect(resolveRegionalReferenceImages({ referenceImages: [ipAdapterRef(null)] }, 'sd-1')).toEqual([]);
  });

  it('keeps an IP-Adapter ref once an image is assigned for a supported SD base', () => {
    const inputs = resolveRegionalReferenceImages({ referenceImages: [ipAdapterRef(asset)] }, 'sd-1');
    expect(inputs).toHaveLength(1);
    expect(inputs[0]).toMatchObject({ id: 'ref-ipa', imageName: 'ref.png', type: 'ip_adapter' });
  });

  it('drops regional reference images for FLUX.2', () => {
    expect(resolveRegionalReferenceImages({ referenceImages: [ipAdapterRef(asset)] }, 'flux2')).toEqual([]);
  });

  it.each(['krea-2', 'z-image', 'anima'])('drops regional reference images for %s', (base) => {
    const ref = ipAdapterRef(asset);
    const matchingBase = { ...ref, config: { ...ref.config, model: { ...ipAdapterModel, base } } };
    expect(resolveRegionalReferenceImages({ referenceImages: [matchingBase] }, base)).toEqual([]);
  });

  it('keeps an SD2 IP-Adapter ref whose model matches the base', () => {
    const sd2Ref = {
      ...ipAdapterRef(asset),
      config: { ...ipAdapterRef(asset).config, model: { ...ipAdapterModel, base: 'sd-2' } },
    };
    expect(resolveRegionalReferenceImages({ referenceImages: [sd2Ref] }, 'sd-2')).toHaveLength(1);
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

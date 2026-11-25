import type { GenerateModelConfig, MainModelConfig } from '@features/generation/contracts';
import type {
  CanvasDocumentContractV3,
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/contracts';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type { WorkbenchState } from '@workbench/projectContracts';
import type { WorkbenchAction } from '@workbench/workbenchState.testing';
import type { WorkbenchCommands } from '@workbench/workbenchStore';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
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
import { createInstance } from 'i18next';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { RunCanvasInvocationDeps } from './prepareCanvasInvocation';

import { DEFAULT_CANVAS_COMPOSITING } from './canvasCompositing';
import { describeCanvasInvocationFailure } from './canvasInvocationFailure';
import { DEFAULT_CANVAS_SCALING } from './canvasScaling';
import { runCanvasInvocation } from './prepareCanvasInvocation';

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

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const english = createInstance();
await english.init({
  initAsync: false,
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: Object.values(englishCatalogModules)[0] as Record<string, unknown> } },
});
const t = english.t.bind(english) as (key: string, options?: Record<string, unknown>) => string;

const makeDoc = (layers: CanvasRasterLayerContractV2[], size = 64): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: size, width: size, x: 0, y: 0 },
  height: size,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
  width: size,
});

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
          // Resolve on a microtask so a missing `await` would let compositing start first — the ordering assertion
          // depends on this.
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
    formatCanvasInvocationFailure: (failure) => describeCanvasInvocationFailure(t, failure),
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

describe('runCanvasInvocation', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('generates txt2img without any composite upload when content does not overlap the bbox', async () => {
    const harness = makeHarness({ document: makeDoc([]) });

    await runCanvasInvocation(harness.deps);

    expect(harness.uploadImage).not.toHaveBeenCalled();
    expect(harness.events).not.toContain('getLayerSurface');

    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.graph.label).toBe('SD 1.5 txt2img');
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

  it('reuses committed composite uploads on a later successful invocation', async () => {
    const harness = makeHarness();

    await runCanvasInvocation(harness.deps);
    await runCanvasInvocation(harness.deps);

    expect(harness.submittedGraphs()).toHaveLength(2);
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
  });

  it('threads a Gallery destination through to the snapshot and a durable (non-intermediate) output', async () => {
    const harness = makeHarness({ destination: 'gallery', document: makeDoc([]) });

    await runCanvasInvocation(harness.deps);

    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.destination).toBe('gallery');
    expect(submitted[0]!.graph.backendGraph!.nodes.canvas_output!.is_intermediate).toBe(false);
  });

  it('awaits the upload flush before compositing', async () => {
    const harness = makeHarness();

    await runCanvasInvocation(harness.deps);

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

  it('generates from text for a text-only provider when the frame holds no visible pixels', async () => {
    const harness = makeHarness({ alpha: 0, model: externalModel });

    await runCanvasInvocation(harness.deps);

    expect(harness.notices()).toEqual([]);
    const submitted = harness.submittedGraphs();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.graph.backendGraph?.nodes.canvas_processing_output).toMatchObject({ mode: 'txt2img' });
    expect(submitted[0]?.graph.backendGraph?.nodes.canvas_processing_output).not.toHaveProperty('init_image');
  });

  it('uses actual frame alpha for a hosted-provider init image budget with identical layer geometry', async () => {
    const model: GenerateModelConfig = {
      ...externalModel,
      name: 'Seedream',
      provider_id: 'seedream',
      provider_model_id: 'seedream-4-5-251128',
      capabilities: { modes: ['txt2img', 'img2img'], supports_reference_images: true, max_reference_images: 14 },
    };
    const generateValues = {
      referenceImages: Array.from({ length: 14 }, (_, i) => ({
        id: `ref-${i}`,
        isEnabled: true,
        config: {
          type: 'external_reference_image',
          image: {
            imageName: `ref-${i}.png`,
            width: 512,
            height: 512,
            original: { image: { image_name: `ref-${i}.png`, width: 512, height: 512 } },
          },
        },
      })),
    };
    const document = makeDoc([rasterLayer('same-layer')]);
    const transparent = makeHarness({ alpha: 0, document, generateValues, model });
    await runCanvasInvocation(transparent.deps);

    expect(transparent.notices()).toEqual([]);
    expect(transparent.submittedGraphs()).toHaveLength(1);
    const output = transparent.submittedGraphs()[0]?.graph.backendGraph?.nodes.canvas_processing_output;
    expect(output).toMatchObject({ mode: 'txt2img' });
    expect(output).not.toHaveProperty('init_image');

    const opaque = makeHarness({ alpha: 255, document, generateValues, model });
    await runCanvasInvocation(opaque.deps);

    expect(opaque.submittedGraphs()).toEqual([]);
    expect(opaque.notices()).toMatchObject([
      { message: 'Seedream supports at most 15 images total (reference + init + output), got 16.' },
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

    it('refuses a layer the judge blocks with a localized notice, before uploading anything', async () => {
      const harness = makeHarness({ document: docWithLayers([typed('depth-layer', 'depth')]) });

      await runCanvasInvocation(harness.deps);

      expect(harness.notices()).toEqual([
        expect.objectContaining({
          message: `Control layer "depth-layer": ${t('widgets.layers.control.validation.models_unavailable')}`,
          title: 'Canvas generation failed',
        }),
      ]);
      expect(harness.submittedGraphs()).toHaveLength(0);
      expect(harness.uploadImage).not.toHaveBeenCalled();
    });
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
      expect(graph?.nodes.anima_lllite_a?.control_model).not.toHaveProperty('cond_in_channels');
      expect(
        graph?.edges.filter((edge) => edge.destination.field === 'control_lllite').map((edge) => edge.source.node_id)
      ).toEqual(['control_lllite_collector']);
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
    const firstRun = runCanvasInvocation(first.deps);
    await Promise.resolve();

    const second = makeHarness({ inFlight });
    await runCanvasInvocation(second.deps);

    expect(second.submittedGraphs()).toHaveLength(0);
    expect(second.notices()).toHaveLength(0);
    expect(second.uploadImage).not.toHaveBeenCalled();

    releaseFlush();
    await firstRun;
    expect(first.submittedGraphs()).toHaveLength(1);
  });

  it('enqueues into the originating project, not the active one, when the active project changes mid-flight', async () => {
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
});

describe('runCanvasInvocation — layout regions on MOSS-Image', () => {
  const mossModel: MainModelConfig = {
    base: 'moss-image',
    format: 'diffusers',
    key: 'moss-model',
    name: 'MOSS-Image',
    type: 'main',
  };
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
});

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
    expect((nodes.create_gradient_mask?.mask as { image_name?: string } | undefined)?.image_name).toBeDefined();
    expect(harness.notices()).toHaveLength(0);
    expect(harness.surfaceIds).toEqual(['base', 'mask']);
  });

  it("sends an inpainting mask's guidance to the graph as a region over the mask", async () => {
    const guided = {
      ...inpaintMaskLayer('mask'),
      guidance: { autoNegative: false, isEnabled: true, negativePrompt: null, positivePrompt: 'a red fox' },
    };
    const harness = makeHarness({ document: docWithLayers([rasterLayer('base'), guided]) });

    await runCanvasInvocation(harness.deps);

    expect(harness.notices()).toHaveLength(0);
    const submitted = harness.submittedGraphs()[0]?.graph;
    expect(submitted?.label).toBe('SD 1.5 inpaint');
    expect(submitted?.backendGraph?.nodes.rg_pos_cond_mask).toMatchObject({ prompt: 'a red fox' });
    expect(submitted?.backendGraph?.nodes.rg_mask_to_tensor_mask).toMatchObject({
      image: { image_name: expect.stringMatching(/^composite-\d+\.png$/) },
      type: 'alpha_mask_to_tensor',
    });
  });

  it('dispatches an outpaint graph when content only partially covers the bbox', async () => {
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
});

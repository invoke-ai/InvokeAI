import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/api';
import type { InvocationRoute, InvocationValidationReason } from '@workbench/invocationContracts';
import type { WidgetId } from '@workbench/widgetContracts';

import { galleryKeys } from '@features/gallery/queries';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { createDefaultUpscaleWidgetValues } from '@features/upscale';
import { queryClient } from '@platform/query/client';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as settingsStoreModule from './settings/store';

// Dynamic prompt expansion is the only transport `submitResolvedInvocation` touches.
const parseDynamicPromptsMock = vi.hoisted(() => vi.fn());

vi.mock('@features/generation/data/promptUtilities', () => ({ parseDynamicPrompts: parseDynamicPromptsMock }));

// Remembered control-model picks, which readiness reads imperatively from the account preferences.
const preferences = vi.hoisted(() => ({ controlModelMemory: {} as Record<string, string> }));

/** A route reason as text: Generate issues carry their English beside the locale key. */
const reasonText = (reason: InvocationValidationReason) =>
  typeof reason === 'object' && 'generationIssue' in reason ? reason.generationIssue.message : reason;

vi.mock('./settings/store', async (importOriginal) => {
  const actual = await importOriginal<typeof settingsStoreModule>();
  return {
    ...actual,
    getWorkbenchPreferences: () => ({
      ...actual.getWorkbenchPreferences(),
      controlModelMemory: preferences.controlModelMemory,
    }),
  };
});

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';

import {
  areInvocationRouteInputsEqual,
  getInvocationRouteInput,
  resolveInvocationRoute,
  resolveInvocationRouteInput,
} from './invocation';
import { submitResolvedInvocation } from './invocationSubmit';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';
import { createWorkbenchStore } from './workbenchStore';

// Production words rejections from the locale; tests only need the structured rejection to reach the notice.
const formatControlLayerError = ({ code, layerName }: { code: string; layerName: string }) => `${layerName}: ${code}`;
const formatUpscaleMetadataFailure = (name: string) => `metadataFailed: ${name}`;

const animaModel: MainModelConfig = { base: 'anima', key: 'anima-model', name: 'Anima', type: 'main' };
const animaVae: VaeModelConfig = { base: 'qwen-image', key: 'anima-vae', name: 'Anima VAE', type: 'vae' };
const qwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen3-encoder',
  name: 'Qwen3 Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
};
const unknownExternalModel: GenerateModelConfig = {
  base: 'external',
  capabilities: { modes: ['txt2img'] },
  format: 'external_api',
  key: 'external-unknown',
  name: 'Unknown External Provider',
  provider_id: 'future-provider',
  type: 'external_image_generator',
};

const upscaleModel = (key: string, type: string, base: string, name = key): ModelConfig => ({
  base,
  file_size: 1,
  format: 'checkpoint',
  hash: `${key}-hash`,
  key,
  name,
  path: key,
  source: key,
  source_type: 'path',
  type,
});
const upscaleModels = [
  upscaleModel('sdxl-upscale', 'main', 'sdxl'),
  upscaleModel('spandrel-upscale', 'spandrel_image_to_image', 'any'),
  upscaleModel('tile-upscale', 'controlnet', 'sdxl', 'SDXL Tile ControlNet'),
];

const createGenerateValues = (
  model: GenerateModelConfig = animaModel,
  overrides: Partial<GenerateWidgetValues> = {}
): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(model),
  model,
  modelKey: model.key,
  positivePrompt: 'an anima prompt',
  ...overrides,
});

const getActiveProject = (values: GenerateWidgetValues, canvasValues?: Record<string, unknown>) => {
  let state = workbenchReducer(createInitialWorkbenchState(), { type: 'setGenerateSettings', values });
  if (canvasValues) {
    state = workbenchReducer(state, { type: 'patchWidgetValues', values: canvasValues, widgetId: 'canvas' });
  }
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  expect(project).toBeDefined();

  return project!;
};

seedArchitectureCapabilities();

describe('resolveInvocationRoute', () => {
  it('invalidates Anima generation until required components are selected', () => {
    const missingComponentsRoute = resolveInvocationRoute(
      getActiveProject(createGenerateValues(animaModel, { vae: animaVae }))
    );

    expect(missingComponentsRoute.sourceValid).toBe(false);
    expect(missingComponentsRoute.validationReasons.map(reasonText)).toContain(
      'Generate needs a Qwen3 Encoder for Anima models.'
    );

    const validRoute = resolveInvocationRoute(
      getActiveProject(createGenerateValues(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: animaVae }))
    );

    expect(validRoute.sourceValid).toBe(true);
    expect(validRoute.validationReasons).toEqual([]);
  });
});

describe('resolveInvocationRoute — upscale source', () => {
  const route: InvocationRoute = {
    destination: 'gallery',
    destinationLocked: false,
    sourceId: 'upscale',
    sourceLocked: false,
  };

  const createInput = () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, {
      type: 'patchWidgetValues',
      values: {
        ...createDefaultUpscaleWidgetValues(upscaleModels),
        inputImage: { height: 512, image_name: 'input.png', width: 768 },
      },
      widgetId: 'upscale',
    });

    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId)!;

    return getInvocationRouteInput(project);
  };

  it('is ready only with a mounted widget and installed compatible required models', () => {
    expect(resolveInvocationRouteInput(createInput(), 'global', route, upscaleModels)).toMatchObject({
      sourceValid: true,
      validationReasons: [],
    });

    const input = createInput();
    const withoutWidget = { ...input, mountedWidgetIds: input.mountedWidgetIds.filter((id) => id !== 'upscale') };
    const missingTile = upscaleModels.filter((model) => model.key !== 'tile-upscale');

    expect(
      resolveInvocationRouteInput(withoutWidget, 'global', route, upscaleModels).validationReasons.map(reasonText)
    ).toContainEqual({ routeIssue: { kind: 'sourceNotMounted', sourceId: 'upscale' } });
    expect(
      resolveInvocationRouteInput(input, 'global', route, missingTile).validationReasons.map(reasonText)
    ).toContain('SDXL Tile ControlNet is no longer installed.');
  });
});

describe('resolveInvocationRoute — canvas source', () => {
  const canvasRoute: InvocationRoute = {
    destination: 'canvas',
    destinationLocked: false,
    sourceId: 'canvas',
    sourceLocked: false,
  };

  const validGenerateValues = createGenerateValues(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: animaVae });

  const canvasInputFor = (
    overrides: {
      generateValues?: typeof validGenerateValues;
      mountedWidgetIds?: WidgetId[];
      bbox?: { width: number; height: number };
      layers?: CanvasLayerContract[];
      rasterInFrame?: boolean;
    } = {}
  ) => {
    const project = getActiveProject(overrides.generateValues ?? validGenerateValues);

    return {
      ...getInvocationRouteInput(project),
      canvasBbox: overrides.bbox ?? { height: 512, width: 512 },
      canvasLayers: overrides.layers ?? [],
      canvasRasterInFrame: overrides.rasterInFrame ?? false,
      mountedWidgetIds: overrides.mountedWidgetIds ?? (['canvas', 'generate'] as WidgetId[]),
    };
  };

  const loadedModels = [animaModel, qwen3Encoder, animaVae] as unknown as ModelConfig[];

  const modellessControlLayer = (source?: CanvasControlLayerContract['source']): CanvasControlLayerContract => ({
    // The fixture's main model is Anima, so the layer carries the kind Anima runs.
    adapter: { beginEndStepPct: [0, 1], controlMode: null, kind: 'anima_lllite', model: null, weight: 1 },
    blendMode: 'normal',
    id: 'control-1',
    isEnabled: true,
    isLocked: false,
    name: 'Control Layer 1',
    opacity: 1,
    source: source ?? { image: { height: 64, imageName: 'control.png', width: 64 }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'control',
    withTransparencyEffect: true,
  });

  it('is a valid source when the widget is mounted, the model is valid, and the frame has area', () => {
    const route = resolveInvocationRouteInput(canvasInputFor(), 'global', canvasRoute);

    expect(route.sourceValid).toBe(true);
    expect(route.validationReasons).toEqual([]);
  });

  it('validates provider dimensions against the Canvas bbox rather than the Generate panel size', () => {
    const model: GenerateModelConfig = {
      ...unknownExternalModel,
      provider_id: 'openai',
      capabilities: { modes: ['txt2img'], max_image_size: { width: 1024, height: 1024 } },
    };
    const generateValues = createGenerateValues(model, { width: 2048, height: 2048 });
    expect(resolveInvocationRouteInput(canvasInputFor({ generateValues }), 'global', canvasRoute).sourceValid).toBe(
      true
    );
    expect(
      resolveInvocationRouteInput(
        canvasInputFor({ generateValues, bbox: { width: 1025, height: 512 } }),
        'global',
        canvasRoute
      ).validationReasons.map(reasonText)
    ).toContain('Unknown External Provider supports at most 1024×1024.');
  });

  it('blocks a text-only provider while raster content overlaps the frame, naming the content', () => {
    const textOnly: GenerateModelConfig = { ...unknownExternalModel, provider_id: 'openai' };
    const editing: GenerateModelConfig = { ...textOnly, capabilities: { modes: ['txt2img', 'img2img'] } };

    expect(
      resolveInvocationRouteInput(
        canvasInputFor({ generateValues: createGenerateValues(textOnly), rasterInFrame: true }),
        'global',
        canvasRoute
      ).validationReasons
    ).toEqual([
      'Unknown External Provider cannot use image content in the generation frame. Hide or move raster content out of the frame to generate.',
    ]);
    expect(
      resolveInvocationRouteInput(
        canvasInputFor({ generateValues: createGenerateValues(textOnly) }),
        'global',
        canvasRoute
      ).sourceValid
    ).toBe(true);
    expect(
      resolveInvocationRouteInput(
        canvasInputFor({ generateValues: createGenerateValues(editing), rasterInFrame: true }),
        'global',
        canvasRoute
      ).sourceValid
    ).toBe(true);
  });

  it('counts the Canvas source image before accepting a hosted-provider reference budget', () => {
    const model: GenerateModelConfig = {
      ...unknownExternalModel,
      provider_id: 'seedream',
      provider_model_id: 'seedream-4-5-251128',
      capabilities: { modes: ['txt2img', 'img2img'], supports_reference_images: true, max_reference_images: 14 },
    };
    const generateValues = createGenerateValues(model, {
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
    });
    const input = canvasInputFor({ generateValues, rasterInFrame: true });
    expect(resolveInvocationRouteInput(input, 'global', canvasRoute).validationReasons.map(reasonText)).toContain(
      'Unknown External Provider supports at most 15 images total (reference + init + output), got 16.'
    );
    expect(
      resolveInvocationRouteInput({ ...input, canvasRasterInFrame: false }, 'global', canvasRoute).sourceValid
    ).toBe(true);
    expect(resolveInvocationRouteInput(input, 'global', { ...canvasRoute, sourceId: 'generate' }).sourceValid).toBe(
      true
    );
  });

  it('blocks a typed content-bearing control layer no model resolves once models are loaded', () => {
    const route = resolveInvocationRouteInput(
      canvasInputFor({ layers: [{ ...modellessControlLayer(), controlType: 'depth', detection: null }] }),
      'global',
      canvasRoute,
      loadedModels
    );

    expect(route.sourceValid).toBe(false);
    // Kept structured so the shell words it from the locale.
    expect(route.validationReasons).toContainEqual({
      controlLayerIssue: {
        code: 'no_control_models',
        layerId: expect.any(String),
        layerName: 'Control Layer 1',
      },
    });
  });

  describe('layout regions', () => {
    const layoutModel = (base: 'moss-image' | 'ideogram-4', name: string): MainModelConfig => ({
      base,
      format: 'diffusers',
      key: `${base}-model`,
      name,
      type: 'main',
    });
    const mossModel = layoutModel('moss-image', 'MOSS-Image');
    const ideogramModel = layoutModel('ideogram-4', 'Ideogram 4');
    // The frame sits away from the document origin, so bounds must be compared in document space.
    const frame = { height: 512, width: 512, x: 256, y: 128 };
    /** A captioned region whose 64×64 mask bitmap starts at the layer's position. */
    const region = (
      id: string,
      position: { x: number; y: number },
      overrides: Partial<CanvasRegionalGuidanceLayerContract> = {}
    ): CanvasRegionalGuidanceLayerContract => ({
      autoNegative: false,
      blendMode: 'normal',
      id,
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: { height: 64, imageName: `${id}.png`, width: 64 }, fill: { color: '#00ff00', style: 'solid' } },
      name: id,
      negativePrompt: null,
      opacity: 1,
      positivePrompt: 'a shop sign',
      referenceImages: [],
      transform: { rotation: 0, scaleX: 1, scaleY: 1, ...position },
      type: 'regional_guidance',
      ...overrides,
    });
    const projectFor = (
      model: MainModelConfig,
      overrides: Partial<GenerateWidgetValues>,
      layers: CanvasRegionalGuidanceLayerContract[],
      sourceId: InvocationRoute['sourceId'] = 'canvas'
    ) => {
      const project = getActiveProject(createGenerateValues(model, overrides));
      return {
        ...project,
        canvas: {
          ...project.canvas,
          document: {
            ...project.canvas.document,
            bbox: frame,
            stacks: { ...project.canvas.document.stacks, regional_guidance: layers },
          },
        },
        invocation: { ...project.invocation, sourceId },
      };
    };
    const inputFor = (...args: Parameters<typeof projectFor>) => getInvocationRouteInput(projectFor(...args));
    const reasonsFor = (
      model: MainModelConfig,
      overrides: Partial<GenerateWidgetValues>,
      layers: CanvasRegionalGuidanceLayerContract[]
    ) =>
      resolveInvocationRouteInput(
        { ...inputFor(model, overrides, layers), mountedWidgetIds: ['canvas', 'generate'] as WidgetId[] },
        'global',
        canvasRoute
      ).validationReasons;
    const blankPromptReason = (label: string) =>
      `${label} needs a prompt to place regional guidance in its layout. Enter a prompt or disable the regional guidance layers.`;

    it.each([
      [mossModel, 'MOSS-Image'],
      [ideogramModel, 'Ideogram 4'],
    ])('blocks a region inside the frame around a blank prompt before anything uploads: %s', (model, label) => {
      expect(reasonsFor(model, { positivePrompt: ' ' }, [region('inside', { x: 300, y: 200 })])).toEqual([
        blankPromptReason(label),
      ]);
      // A prompt template supplies the prompt the layout is built around.
      const template = { id: 'poster', name: 'Poster', negativePrompt: '', positivePrompt: 'a vintage poster' };
      expect(
        reasonsFor(model, { positivePrompt: '', promptTemplate: template }, [region('inside', { x: 300, y: 200 })])
      ).toEqual([]);
    });
  });

  it('treats the layer stack identity as part of the route input', () => {
    const input = canvasInputFor();

    expect(areInvocationRouteInputsEqual(input, { ...input })).toBe(true);
    expect(areInvocationRouteInputsEqual(input, { ...input, canvasLayers: [...input.canvasLayers] })).toBe(false);
  });
});

/** A fresh listing of every readable board, as `ensureGalleryBoardDestinations` leaves it: the boards named are destinations. */
const listGalleryBoardsInCache = (...boardIds: string[]) =>
  queryClient.setQueryData(
    galleryKeys.boards(captureAccountScope(), {
      includeArchived: true,
      includeDateBoards: false,
      orderBy: 'created_at',
      orderDir: 'DESC',
    }),
    boardIds.map((id) => ({
      archived: false,
      assetCount: 0,
      assetVideoCount: 0,
      id,
      imageCount: 0,
      isInbox: false,
      kind: 'board' as const,
      name: id,
      projectId: null,
      videoCount: 0,
    }))
  );

describe('submitResolvedInvocation', () => {
  const routeFor = (project: Parameters<typeof resolveInvocationRoute>[0], route: InvocationRoute) =>
    resolveInvocationRoute(project, 'global', route);

  beforeEach(() => {
    queryClient.clear();
    listGalleryBoardsInCache();
    parseDynamicPromptsMock.mockReset();
    parseDynamicPromptsMock.mockResolvedValue({ prompts: ['a red cat', 'a green cat'] });
  });

  it('routes a canvas source through prepareCanvasInvocation and does not dispatch a resolved snapshot', () => {
    const project = getActiveProject(
      createGenerateValues(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: animaVae }),
      { outputOnlyMaskedRegions: false }
    );
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();
    const route = routeFor(project, { ...project.invocation, destination: 'gallery', sourceId: 'canvas' });
    const owner = captureAccountScope();

    submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner,
      prepareCanvasInvocation,
      project,
      route,
    });

    expect(submitResolved).not.toHaveBeenCalled();
    expect(prepareCanvasInvocation).toHaveBeenCalledTimes(1);
    // The resolved destination rides through so a Canvas source can target the Gallery.
    expect(prepareCanvasInvocation.mock.calls[0]?.[0]).toMatchObject({
      destination: 'gallery',
      compositing: expect.objectContaining({ outputOnlyMaskedRegions: false }),
      owner,
      projectId: project.id,
    });
  });

  it('settles only after canvas preparation finishes so the active invoke guard stays held', async () => {
    const project = getActiveProject(
      createGenerateValues(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: animaVae })
    );
    const commands = createWorkbenchStore().commands;
    let releasePreparation = (): void => undefined;
    const preparation = new Promise<void>((resolve) => {
      releasePreparation = resolve;
    });
    const prepareCanvasInvocation = vi.fn(() => preparation);
    const route = routeFor(project, { ...project.invocation, destination: 'canvas', sourceId: 'canvas' });
    let settled = false;

    const submission = submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation,
      project,
      route,
    }).then(() => {
      settled = true;
    });

    expect(prepareCanvasInvocation).toHaveBeenCalledTimes(1);
    await Promise.resolve();
    expect(settled).toBe(false);

    releasePreparation();
    await submission;
    expect(settled).toBe(true);
  });

  it('expands a dynamic prompt before dispatching, and only then', async () => {
    const project = getActiveProject(
      createGenerateValues(animaModel, {
        positivePrompt: 'a {red|green} cat',
        qwen3EncoderModel: qwen3Encoder,
        vae: animaVae,
      })
    );
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const route = routeFor(project, { ...project.invocation, destination: 'gallery', sourceId: 'generate' });

    submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      project,
      route,
    });

    // The expansion is a round trip, so dispatch cannot have happened yet.
    expect(submitResolved).not.toHaveBeenCalled();

    await vi.waitFor(() => expect(submitResolved).toHaveBeenCalledTimes(1));
    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject({
      positivePrompts: ['a red cat', 'a green cat'],
    });
  });

  it('dispatches synchronously when the prompt has no dynamic syntax', () => {
    const project = getActiveProject(createGenerateValues(animaModel, { positivePrompt: 'a plain cat' }));
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const route = routeFor(project, { ...project.invocation, destination: 'gallery', sourceId: 'generate' });

    submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      project,
      route,
    });

    expect(submitResolved).toHaveBeenCalledTimes(1);
    expect(submitResolved.mock.calls[0]?.[0]).not.toHaveProperty('positivePrompts');
  });

  // Expansion failures must not submit literal dynamic syntax.
  describe('when a prompt cannot be expanded', () => {
    const submitDynamicPrompt = () => {
      const project = getActiveProject(createGenerateValues(animaModel, { positivePrompt: 'a {red|green} cat' }));
      const store = createWorkbenchStore();
      const submitResolved = vi.spyOn(store.commands.generation, 'submitResolved');
      const route = routeFor(project, { ...project.invocation, destination: 'gallery', sourceId: 'generate' });

      submitResolvedInvocation({
        commands: store.commands,
        formatControlLayerError,
        formatUpscaleMetadataFailure,
        scopedDefaults: {},
        models: undefined,
        owner: captureAccountScope(),
        prepareCanvasInvocation: vi.fn(),
        project,
        route,
      });

      return { store, submitResolved };
    };

    it('does not dispatch when the request fails outright', async () => {
      parseDynamicPromptsMock.mockRejectedValueOnce(new Error('offline'));

      const { store, submitResolved } = submitDynamicPrompt();

      await vi.waitFor(() => expect(store.getSnapshot().notifications).toHaveLength(1));
      expect(submitResolved).not.toHaveBeenCalled();
      // No server reason came back, so the notice says the server could not be reached.
      expect(store.getSnapshot().notifications[0]).toMatchObject({
        kind: 'error',
        messageKey: 'topbar.invoke.expansionUnreachable',
        title: 'The prompt could not be expanded',
        titleKey: 'topbar.invoke.expansionFailed',
      });
    });
  });

  it('quietly ignores a submission owned by an expired account scope', () => {
    const project = getActiveProject(createGenerateValues(animaModel));
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();
    const route = routeFor(project, { ...project.invocation, sourceId: 'generate' });
    const owner = captureAccountScope();

    accountLifecycle.invalidate();
    submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner,
      prepareCanvasInvocation,
      project,
      route,
    });

    expect(prepareCanvasInvocation).not.toHaveBeenCalled();
    expect(submitResolved).not.toHaveBeenCalled();
  });
});

describe('resolveInvocationRoute — video destination compatibility', () => {
  it('blocks the canvas destination for a video source and allows the gallery', () => {
    const state = createInitialWorkbenchState();
    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId)!;
    const input = getInvocationRouteInput(project);
    const canvasRoute: InvocationRoute = {
      destination: 'canvas',
      destinationLocked: false,
      sourceId: 'video',
      sourceLocked: false,
    };

    // Canvas cannot stage video, and only Gallery-destined videos are boarded.
    const blocked = resolveInvocationRouteInput(input, 'global', canvasRoute);

    expect(blocked.destinationValid).toBe(false);
    expect(blocked.validationReasons).toContainEqual({ routeIssue: { kind: 'videoNeedsGallery' } });

    expect(
      resolveInvocationRouteInput(input, 'global', { ...canvasRoute, destination: 'gallery' }).destinationValid
    ).toBe(true);
  });
});

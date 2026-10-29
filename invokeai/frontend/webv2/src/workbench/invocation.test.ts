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
  CanvasNodeContract,
  CanvasRasterLayerContractV2,
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
const flux2Model: MainModelConfig = {
  base: 'flux2',
  format: 'gguf_quantized',
  key: 'flux2-klein-9b',
  name: 'FLUX.2 Klein 9B',
  type: 'main',
  variant: 'klein_9b',
};
const incompatibleFlux2Source: MainModelConfig = {
  base: 'flux2',
  format: 'diffusers',
  key: 'flux2-klein-4b-source',
  name: 'FLUX.2 Klein 4B Source',
  type: 'main',
  variant: 'klein_4b',
};
const flux2Vae: VaeModelConfig = { base: 'flux2', key: 'flux2-vae', name: 'FLUX.2 VAE', type: 'vae' };
const flux2Qwen3Encoder: ComponentModelConfig = {
  base: 'any',
  key: 'flux2-qwen3-encoder',
  name: 'Qwen3 8B Encoder',
  type: 'qwen3_encoder',
  variant: 'qwen3_8b',
};
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

  it('invalidates non-Diffusers FLUX.2 generation when a stale component source cannot provide the required Qwen3 encoder', () => {
    const route = resolveInvocationRoute(
      getActiveProject(
        createGenerateValues(flux2Model, { componentSourceModel: incompatibleFlux2Source, vae: flux2Vae })
      )
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons.map(reasonText)).toContain(
      'Generate needs a Qwen3 Encoder for non-Diffusers FLUX.2 models.'
    );
  });

  it('invalidates FLUX.2 generation when dimensions are not on the required 16px grid', () => {
    const route = resolveInvocationRoute(
      getActiveProject(
        createGenerateValues(flux2Model, { height: 888, qwen3EncoderModel: flux2Qwen3Encoder, vae: flux2Vae })
      )
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons.map(reasonText)).toContain('Height must be a multiple of 16.');
  });

  it('invalidates external providers without a registered invocation node', () => {
    const route = resolveInvocationRoute(getActiveProject(createGenerateValues(unknownExternalModel)));

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons.map(reasonText)).toContain(
      "No invocation node registered for external provider 'future-provider'."
    );
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

  it('still counts the widget as mounted once it is floated out of its rail', () => {
    let state = createInitialWorkbenchState();

    state = workbenchReducer(state, { instanceId: 'upscale', type: 'floatWidget' });

    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId)!;

    expect(project.floatingWidgets?.upscale).toBeDefined();
    expect(getInvocationRouteInput(project).mountedWidgetIds).toContain('upscale');
  });

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
    ).toContain('The Upscale widget is not mounted in this project.');
    expect(
      resolveInvocationRouteInput(input, 'global', route, missingTile).validationReasons.map(reasonText)
    ).toContain('SDXL Tile ControlNet is no longer installed.');
  });

  it('rejects a main-model base the widget has no builder for', () => {
    // FLUX.1 used to stand in for "unsupported" here; it is supported now, so this needs a base
    // that genuinely has no entry in UPSCALE_ARCHITECTURES.
    const input = createInput();
    const values = { ...input.upscaleValues, model: upscaleModel('sd3-upscale', 'main', 'sd-3', 'SD3') };
    const models = [values.model, upscaleModels[1]!];
    const result = resolveInvocationRouteInput({ ...input, upscaleValues: values }, 'global', route, models);

    expect(result.sourceValid).toBe(false);
    expect(result.validationReasons.map(reasonText)).toContain(
      'Upscale supports only SD1.5, SDXL, FLUX.1, FLUX.2, Z-Image, Anima, Krea-2 or Qwen Image main models.'
    );
  });

  it('rejects a ControlNet that is not a Tile model for its base', () => {
    const input = createInput();
    const values = {
      ...input.upscaleValues,
      tileControlnetModel: upscaleModel('depth-upscale', 'controlnet', 'sdxl', 'SDXL Depth ControlNet'),
    };
    const models = [...upscaleModels, values.tileControlnetModel];
    const result = resolveInvocationRouteInput({ ...input, upscaleValues: values }, 'global', route, models);

    expect(result.sourceValid).toBe(false);
    expect(result.validationReasons.map(reasonText)).toContain(
      'The Tile ControlNet must match the main model base and be a Tile or Union model.'
    );
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

  it('allows Canvas preparation for an img2img-only provider while Generate still requires txt2img', () => {
    const model: GenerateModelConfig = {
      ...unknownExternalModel,
      provider_id: 'openai',
      capabilities: { modes: ['img2img'] },
    };
    const input = canvasInputFor({ generateValues: createGenerateValues(model) });
    expect(resolveInvocationRouteInput(input, 'global', canvasRoute).sourceValid).toBe(true);
    expect(
      resolveInvocationRouteInput(input, 'global', { ...canvasRoute, sourceId: 'generate' }).validationReasons.map(
        reasonText
      )
    ).toContain('Unknown External Provider does not support text-to-image generation.');
  });

  it('names unsupported provider control layers in readiness even without a loaded adapter catalog', () => {
    const model: GenerateModelConfig = { ...unknownExternalModel, provider_id: 'openai' };
    const layer = modellessControlLayer();
    const input = canvasInputFor({ generateValues: createGenerateValues(model), layers: [layer] });
    expect(resolveInvocationRouteInput(input, 'global', canvasRoute).validationReasons).toEqual([
      'Unknown External Provider does not support control layer "Control Layer 1". Disable it to generate.',
    ]);
    expect(
      resolveInvocationRouteInput({ ...input, canvasLayers: [{ ...layer, isEnabled: false }] }, 'global', canvasRoute)
        .sourceValid
    ).toBe(true);
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

  it('validates a fixed-size provider against the size it is asked for, not the frame', () => {
    const model: GenerateModelConfig = {
      ...unknownExternalModel,
      provider_id: 'openai',
      capabilities: {
        aspect_ratio_sizes: { '1:1': { height: 1024, width: 1024 } },
        max_image_size: { height: 1024, width: 1024 },
        modes: ['txt2img'],
      },
    };
    const input = canvasInputFor({ bbox: { height: 3000, width: 3000 }, generateValues: createGenerateValues(model) });

    expect(resolveInvocationRouteInput(input, 'global', canvasRoute).validationReasons).toEqual([]);
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

  it('measures each visible raster layer against the frame, for a text-only provider on a Canvas route', () => {
    const textOnly: GenerateModelConfig = { ...unknownExternalModel, provider_id: 'openai' };
    let project = getActiveProject(createGenerateValues(textOnly));
    const raster = (
      x: number,
      y: number,
      overrides: Partial<CanvasRasterLayerContractV2> = {}
    ): CanvasRasterLayerContractV2 => ({
      blendMode: 'normal',
      id: `raster-${x}-${y}`,
      isEnabled: true,
      isLocked: false,
      name: `Raster ${x},${y}`,
      opacity: 1,
      source: { image: { height: 64, imageName: 'raster.png', width: 64 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y },
      type: 'raster',
      ...overrides,
    });
    const withRaster = (layers: CanvasNodeContract[], sourceId: InvocationRoute['sourceId'] = 'canvas') =>
      getInvocationRouteInput({
        ...project,
        canvas: {
          ...project.canvas,
          document: {
            ...project.canvas.document,
            bbox: { height: 100, width: 100, x: 200, y: 0 },
            stacks: { ...project.canvas.document.stacks, raster: layers },
          },
        },
        invocation: { ...project.invocation, sourceId },
      }).canvasRasterInFrame;

    expect(withRaster([])).toBe(false);
    expect(withRaster([raster(250, 50)])).toBe(true);
    // Layers on either side of an empty frame: their union spans it, but neither reaches it.
    expect(withRaster([raster(0, 0), raster(400, 200)])).toBe(false);
    // Touching the frame's edge is not overlap; hidden or fully transparent content contributes nothing.
    expect(withRaster([raster(300, 0)])).toBe(false);
    expect(withRaster([raster(250, 50, { isEnabled: false })])).toBe(false);
    expect(withRaster([raster(250, 50, { opacity: 0 })])).toBe(false);
    expect(
      withRaster([
        {
          children: [raster(250, 50)],
          id: 'group',
          isEnabled: true,
          isLocked: false,
          name: 'Group',
          opacity: 0,
          type: 'group',
        },
      ])
    ).toBe(false);
    // Readiness only estimates a text layer's extent, so it leaves text to the invoke-time pixel check.
    expect(
      withRaster([
        raster(250, 50, {
          source: {
            align: 'left',
            color: '#000',
            content: 'Hello',
            fontFamily: 'Inter',
            fontSize: 32,
            fontWeight: 400,
            lineHeight: 1.2,
            type: 'text',
          },
        }),
      ])
    ).toBe(false);

    // Other routes and other models never measure.
    expect(withRaster([raster(250, 50)], 'generate')).toBe(false);
    project = getActiveProject(validGenerateValues);
    expect(withRaster([raster(250, 50)])).toBe(false);
  });

  it('invalidates a zero-area generation frame', () => {
    const route = resolveInvocationRouteInput(
      canvasInputFor({ bbox: { height: 0, width: 512 } }),
      'global',
      canvasRoute
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons.map(reasonText)).toContain('Canvas generation frame must have a positive area.');
  });

  it('reuses the generate model validation reasons', () => {
    const route = resolveInvocationRouteInput(
      canvasInputFor({ generateValues: createGenerateValues(animaModel, { vae: animaVae }) }),
      'global',
      canvasRoute
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons.map(reasonText)).toContain('Generate needs a Qwen3 Encoder for Anima models.');
  });

  it('invalidates when the canvas widget is not mounted', () => {
    const route = resolveInvocationRouteInput(
      canvasInputFor({ mountedWidgetIds: ['generate'] as WidgetId[] }),
      'global',
      canvasRoute
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons.map(reasonText)).toContain('The Canvas widget is not mounted in this project.');
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

  it('does not block on a content-bearing layer with neither a type nor a model', () => {
    const route = resolveInvocationRouteInput(
      canvasInputFor({ layers: [modellessControlLayer()] }),
      'global',
      canvasRoute,
      loadedModels
    );

    expect(route.validationReasons).toEqual([]);
  });

  it('re-evaluates a choice between control models against the remembered pick', () => {
    const lllite = (key: string) =>
      ({ base: 'anima', cond_in_channels: 3, key, name: key, type: 'controlnet' }) as unknown as ModelConfig;
    const models = [...loadedModels, lllite('anima-depth-a'), lllite('anima-depth-b')];
    const input = canvasInputFor({ layers: [{ ...modellessControlLayer(), controlType: 'depth', detection: null }] });
    const codes = () =>
      resolveInvocationRouteInput(input, 'global', canvasRoute, models).validationReasons.map((reason) =>
        typeof reason === 'object' && 'controlLayerIssue' in reason ? reason.controlLayerIssue.code : reason
      );

    try {
      expect(codes()).toEqual(['choose_model']);
      preferences.controlModelMemory = { 'anima:depth': 'anima-depth-b' };
      expect(codes()).toEqual([]);
    } finally {
      preferences.controlModelMemory = {};
    }
  });

  it('does not block content-less control layers or check layers before models load', () => {
    const emptyLayerRoute = resolveInvocationRouteInput(
      canvasInputFor({ layers: [modellessControlLayer({ bitmap: null, type: 'paint' })] }),
      'global',
      canvasRoute,
      loadedModels
    );
    expect(emptyLayerRoute.validationReasons).toEqual([]);

    const modelsLoadingRoute = resolveInvocationRouteInput(
      canvasInputFor({ layers: [modellessControlLayer()] }),
      'global',
      canvasRoute
    );
    expect(modelsLoadingRoute.validationReasons).toEqual([]);
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

  it('dispatches submitResolvedInvocationSnapshot for a non-canvas source and never prepares the canvas', () => {
    const project = getActiveProject(
      createGenerateValues(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: animaVae })
    );
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();
    const route = routeFor(project, { ...project.invocation, destination: 'gallery', sourceId: 'generate' });

    submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation,
      project,
      route,
    });

    expect(prepareCanvasInvocation).not.toHaveBeenCalled();
    expect(submitResolved).toHaveBeenCalledTimes(1);
    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject({
      route: expect.objectContaining({ destination: 'gallery', sourceId: 'generate' }),
    });
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
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      project,
      route,
    });

    expect(submitResolved).toHaveBeenCalledTimes(1);
    expect(submitResolved.mock.calls[0]?.[0]).not.toHaveProperty('positivePrompts');
  });

  it('dispatches the prompt as typed, with no round trip, while dynamic prompts are off', () => {
    const project = getActiveProject(
      createGenerateValues(animaModel, { dynamicPromptsEnabled: false, positivePrompt: 'wall painted #ff0000 {a|b}' })
    );
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const route = routeFor(project, { ...project.invocation, destination: 'gallery', sourceId: 'generate' });

    submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
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

    it('does not dispatch when the backend reports an unresolvable wildcard', async () => {
      // The route is deliberately soft: it answers 200 with the literal prompt
      // alongside the message, so only `error` distinguishes this from success.
      parseDynamicPromptsMock.mockResolvedValueOnce({
        error: 'No values found for wildcard(s): nope',
        prompts: ['a {red|green} cat'],
      });

      const { store, submitResolved } = submitDynamicPrompt();

      await vi.waitFor(() => expect(store.getSnapshot().notifications).toHaveLength(1));
      expect(submitResolved).not.toHaveBeenCalled();
      expect(store.getSnapshot().notifications[0]?.message).toBe('No values found for wildcard(s): nope');
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
    expect(blocked.validationReasons.map(reasonText)).toContain(
      'Video results can only be sent to the Gallery. Switch the destination to Gallery.'
    );

    expect(
      resolveInvocationRouteInput(input, 'global', { ...canvasRoute, destination: 'gallery' }).destinationValid
    ).toBe(true);
  });
});

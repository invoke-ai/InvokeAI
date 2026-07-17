import { describe, expect, it, vi } from 'vitest';

import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
} from './generation/types';
import type { ModelConfig } from './models/types';
import type { InvocationRoute, WidgetId } from './types';

import { getDefaultGenerateSettings } from './generation/baseGenerationPolicies';
import { getInvocationRouteInput, resolveInvocationRoute, resolveInvocationRouteInput } from './invocation';
import { submitResolvedInvocation } from './invocationSubmit';
import { createDefaultUpscaleWidgetValues } from './upscale/settings';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState';

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

const getActiveProject = (values: GenerateWidgetValues) => {
  const state = workbenchReducer(createInitialWorkbenchState(), { type: 'setGenerateSettings', values });
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  expect(project).toBeDefined();

  return project!;
};

describe('resolveInvocationRoute', () => {
  it('invalidates Anima generation until required components are selected', () => {
    const missingComponentsRoute = resolveInvocationRoute(
      getActiveProject(createGenerateValues(animaModel, { vae: animaVae }))
    );

    expect(missingComponentsRoute.sourceValid).toBe(false);
    expect(missingComponentsRoute.validationReasons).toContain('Generate needs a Qwen3 Encoder for Anima models.');

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
    expect(route.validationReasons).toContain('Generate needs a Qwen3 Encoder for non-Diffusers FLUX.2 models.');
  });

  it('invalidates FLUX.2 generation when dimensions are not on the required 16px grid', () => {
    const route = resolveInvocationRoute(
      getActiveProject(
        createGenerateValues(flux2Model, { height: 888, qwen3EncoderModel: flux2Qwen3Encoder, vae: flux2Vae })
      )
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons).toContain('Generate height must be a multiple of 16.');
  });

  it('invalidates external providers without a registered invocation node', () => {
    const route = resolveInvocationRoute(getActiveProject(createGenerateValues(unknownExternalModel)));

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons).toContain("No invocation node registered for external provider 'future-provider'.");
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

    expect(resolveInvocationRouteInput(withoutWidget, 'global', route, upscaleModels).validationReasons).toContain(
      'The Upscale widget is not mounted in this project.'
    );
    expect(resolveInvocationRouteInput(input, 'global', route, missingTile).validationReasons).toContain(
      'SDXL Tile ControlNet is no longer installed.'
    );
  });

  it('rejects unsupported main-model bases and non-Tile ControlNets', () => {
    const input = createInput();
    const values = {
      ...input.upscaleValues,
      model: upscaleModel('flux-upscale', 'main', 'flux', 'FLUX'),
      tileControlnetModel: upscaleModel('depth-upscale', 'controlnet', 'flux', 'FLUX Depth ControlNet'),
    };
    const models = [values.model, values.tileControlnetModel, upscaleModels[1]!];
    const result = resolveInvocationRouteInput({ ...input, upscaleValues: values }, 'global', route, models);

    expect(result.sourceValid).toBe(false);
    expect(result.validationReasons).toContain('Upscale supports only SD1.5 and SDXL main models.');
    expect(result.validationReasons).toContain(
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
    } = {}
  ) => {
    const project = getActiveProject(overrides.generateValues ?? validGenerateValues);

    return {
      ...getInvocationRouteInput(project),
      canvasBbox: overrides.bbox ?? { height: 512, width: 512 },
      mountedWidgetIds: overrides.mountedWidgetIds ?? (['canvas', 'generate'] as WidgetId[]),
    };
  };

  it('is a valid source when the widget is mounted, the model is valid, and the frame has area', () => {
    const route = resolveInvocationRouteInput(canvasInputFor(), 'global', canvasRoute);

    expect(route.sourceValid).toBe(true);
    expect(route.validationReasons).toEqual([]);
  });

  it('invalidates a zero-area generation frame', () => {
    const route = resolveInvocationRouteInput(
      canvasInputFor({ bbox: { height: 0, width: 512 } }),
      'global',
      canvasRoute
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons).toContain('Canvas generation frame must have a positive area.');
  });

  it('reuses the generate model validation reasons', () => {
    const route = resolveInvocationRouteInput(
      canvasInputFor({ generateValues: createGenerateValues(animaModel, { vae: animaVae }) }),
      'global',
      canvasRoute
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons).toContain('Generate needs a Qwen3 Encoder for Anima models.');
  });

  it('invalidates when the canvas widget is not mounted', () => {
    const route = resolveInvocationRouteInput(
      canvasInputFor({ mountedWidgetIds: ['generate'] as WidgetId[] }),
      'global',
      canvasRoute
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons).toContain('The Canvas widget is not mounted in this project.');
  });
});

describe('submitResolvedInvocation', () => {
  const routeFor = (project: Parameters<typeof resolveInvocationRoute>[0], route: InvocationRoute) =>
    resolveInvocationRoute(project, 'global', route);

  it('routes a canvas source through prepareCanvasInvocation and does not dispatch a resolved snapshot', () => {
    const project = getActiveProject(
      createGenerateValues(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: animaVae })
    );
    const dispatch = vi.fn();
    const prepareCanvasInvocation = vi.fn();
    const route = routeFor(project, { ...project.invocation, destination: 'gallery', sourceId: 'canvas' });

    submitResolvedInvocation({ dispatch, models: undefined, prepareCanvasInvocation, project, route });

    expect(dispatch).not.toHaveBeenCalled();
    expect(prepareCanvasInvocation).toHaveBeenCalledTimes(1);
    // The resolved destination rides through so a Canvas source can target the Gallery.
    expect(prepareCanvasInvocation.mock.calls[0]?.[0]).toMatchObject({
      destination: 'gallery',
      projectId: project.id,
    });
  });

  it('dispatches submitResolvedInvocationSnapshot for a non-canvas source and never prepares the canvas', () => {
    const project = getActiveProject(
      createGenerateValues(animaModel, { qwen3EncoderModel: qwen3Encoder, vae: animaVae })
    );
    const dispatch = vi.fn();
    const prepareCanvasInvocation = vi.fn();
    const route = routeFor(project, { ...project.invocation, destination: 'gallery', sourceId: 'generate' });

    submitResolvedInvocation({ dispatch, models: undefined, prepareCanvasInvocation, project, route });

    expect(prepareCanvasInvocation).not.toHaveBeenCalled();
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch.mock.calls[0]?.[0]).toMatchObject({
      route: expect.objectContaining({ destination: 'gallery', sourceId: 'generate' }),
      type: 'submitResolvedInvocationSnapshot',
    });
  });
});

import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { InvocationRoute } from '@workbench/invocationContracts';
import type { WidgetId } from '@workbench/widgetContracts';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { createDefaultUpscaleWidgetValues } from '@features/upscale';
import { queryClient } from '@platform/query/client';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// Dynamic prompt expansion is the only transport `submitResolvedInvocation` touches.
const parseDynamicPromptsMock = vi.hoisted(() => vi.fn());

vi.mock('@features/generation/data/promptUtilities', () => ({ parseDynamicPrompts: parseDynamicPromptsMock }));

import {
  areInvocationRouteInputsEqual,
  getInvocationRouteInput,
  resolveInvocationRoute,
  resolveInvocationRouteInput,
} from './invocation';
import { submitResolvedInvocation } from './invocationSubmit';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';
import { createWorkbenchStore } from './workbenchStore';

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
      layers?: CanvasLayerContract[];
    } = {}
  ) => {
    const project = getActiveProject(overrides.generateValues ?? validGenerateValues);

    return {
      ...getInvocationRouteInput(project),
      canvasBbox: overrides.bbox ?? { height: 512, width: 512 },
      canvasLayers: overrides.layers ?? [],
      mountedWidgetIds: overrides.mountedWidgetIds ?? (['canvas', 'generate'] as WidgetId[]),
    };
  };

  const loadedModels = [animaModel, qwen3Encoder, animaVae] as unknown as ModelConfig[];

  const modellessControlLayer = (source?: CanvasControlLayerContract['source']): CanvasControlLayerContract => ({
    adapter: { beginEndStepPct: [0, 1], controlMode: null, kind: 'controlnet', model: null, weight: 1 },
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

  it('blocks a content-bearing control layer with no model once models are loaded', () => {
    const route = resolveInvocationRouteInput(
      canvasInputFor({ layers: [modellessControlLayer()] }),
      'global',
      canvasRoute,
      loadedModels
    );

    expect(route.sourceValid).toBe(false);
    expect(route.validationReasons).toContain('Control layer "Control Layer 1" has no control model selected.');
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

describe('submitResolvedInvocation', () => {
  const routeFor = (project: Parameters<typeof resolveInvocationRoute>[0], route: InvocationRoute) =>
    resolveInvocationRoute(project, 'global', route);

  beforeEach(() => {
    queryClient.clear();
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

    submitResolvedInvocation({ commands, models: undefined, owner, prepareCanvasInvocation, project, route });

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
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      project,
      route,
    });

    expect(submitResolved).toHaveBeenCalledTimes(1);
    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject({ positivePrompts: undefined });
  });

  // Submitting the authored text on a failed expansion would put the literal
  // `{a|b}` or `__name__` in front of the model, so neither failure generates.
  describe('when a prompt cannot be expanded', () => {
    const submitDynamicPrompt = () => {
      const project = getActiveProject(createGenerateValues(animaModel, { positivePrompt: 'a {red|green} cat' }));
      const store = createWorkbenchStore();
      const submitResolved = vi.spyOn(store.commands.generation, 'submitResolved');
      const route = routeFor(project, { ...project.invocation, destination: 'gallery', sourceId: 'generate' });

      submitResolvedInvocation({
        commands: store.commands,
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
      expect(store.getSnapshot().notifications[0]).toMatchObject({
        kind: 'error',
        title: 'The prompt could not be expanded',
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
    submitResolvedInvocation({ commands, models: undefined, owner, prepareCanvasInvocation, project, route });

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

    // A canvas-routed video run would complete with the output appearing
    // nowhere (staging is image-based; only gallery-destined videos board).
    const blocked = resolveInvocationRouteInput(input, 'global', canvasRoute);

    expect(blocked.destinationValid).toBe(false);
    expect(blocked.validationReasons).toContain(
      'Video results can only be sent to the Gallery. Switch the destination to Gallery.'
    );

    expect(
      resolveInvocationRouteInput(input, 'global', { ...canvasRoute, destination: 'gallery' }).destinationValid
    ).toBe(true);
  });
});

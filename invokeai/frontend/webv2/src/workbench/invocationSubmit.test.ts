import type { MainModelConfig } from '@features/generation/contracts';
import type { InvocationRoute } from '@workbench/invocationContracts';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { dynamicPromptsKeys } from '@features/generation/prompts';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { createDefaultUpscaleWidgetValues } from '@features/upscale';
import { queryClient } from '@platform/query/client';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const templatesMock = vi.hoisted(() => ({
  snapshot: { error: null, status: 'idle', templates: {} } as {
    error: string | null;
    status: 'idle' | 'loading' | 'loaded' | 'error';
    templates: Record<string, unknown>;
  },
}));
const generatorsMock = vi.hoisted(() => ({ resolveWorkflowGenerators: vi.fn() }));
// A fresh listing is cached unless a test says otherwise: nothing to await, so dispatches stay synchronous.
const galleryQueriesMock = vi.hoisted(() => ({
  ensureGalleryBoardDestinations: vi.fn((): Promise<void> | null => null),
}));
const galleryMock = vi.hoisted(() => ({
  metadata: vi.fn((_imageName: string, _signal?: AbortSignal): Promise<Record<string, unknown> | null> =>
    Promise.resolve(null)
  ),
}));
vi.mock('@features/gallery', async (importOriginal) => {
  const original = await importOriginal<{ galleryImages: Record<string, unknown> }>();

  return { ...original, galleryImages: { ...original.galleryImages, metadata: galleryMock.metadata } };
});
vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGalleryBoardDestinations: galleryQueriesMock.ensureGalleryBoardDestinations,
}));

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getInvocationTemplatesSnapshot: () => templatesMock.snapshot,
}));
vi.mock('@features/workflow/generators', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  resolveWorkflowGenerators: generatorsMock.resolveWorkflowGenerators,
}));

import { resolveInvocationRoute } from './invocation';
import { submitResolvedInvocation } from './invocationSubmit';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

const getActiveProject = (state: ReturnType<typeof createInitialWorkbenchState>) => {
  const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

  if (!project) {
    throw new Error('No active project');
  }

  return project;
};
import { createWorkbenchStore } from './workbenchStore';

// Production words rejections from the locale; tests only need the structured rejection to reach the notice.
const formatControlLayerError = ({ code, layerName }: { code: string; layerName: string }) => `${layerName}: ${code}`;
const formatUpscaleMetadataFailure = (name: string) => `metadataFailed: ${name}`;

const field = (name: string, typeName: string, overrides: Record<string, unknown> = {}) => ({
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name,
  options: null,
  required: true,
  title: name,
  type: { batch: false, cardinality: 'SINGLE', name: typeName },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});
const template = (type: string, inputs: Record<string, unknown>) => ({
  category: 'test',
  classification: 'stable',
  description: '',
  inputs,
  nodePack: 'invokeai',
  outputs: {},
  outputType: `${type}_output`,
  tags: [],
  title: type,
  type,
  useCache: true,
  version: '1.0.0',
});
const templates = {
  string: template('string', { value: field('value', 'StringField', { default: '' }) }),
  string_batch: template('string_batch', {
    batch_group_id: field('batch_group_id', 'EnumField', { default: 'None', options: ['None'] }),
    strings: field('strings', 'StringField', {
      default: [],
      type: { batch: true, cardinality: 'COLLECTION', name: 'StringField' },
    }),
  }),
  string_generator: template('string_generator', {
    generator: field('generator', 'StringGeneratorField', { input: 'direct' }),
  }),
};
const generatorValue = {
  input: 'a {red|green} cat',
  maxPrompts: 4,
  type: 'string_generator_dynamic_prompts_combinatorial',
};
const invocationNode = (id: string, type: string, inputs: Record<string, unknown>) => ({
  data: {
    inputs: Object.fromEntries(Object.entries(inputs).map(([name, value]) => [name, { label: '', name, value }])),
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type,
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x: 0, y: 0 },
  type: 'invocation' as const,
});
const edge = (source: string, sourceHandle: string, target: string, targetHandle: string) => ({
  id: `${source}->${target}`,
  source,
  sourceHandle,
  target,
  targetHandle,
  type: 'default' as const,
});

/** A workflow whose only batch is fed by a dynamic-prompt generator, so submission must resolve it first. */
const buildProject = () => {
  templatesMock.snapshot = { error: null, status: 'loaded', templates };

  let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'automate', type: 'applyPreset' });

  state = workbenchReducer(state, {
    action: { node: invocationNode('gen', 'string_generator', { generator: generatorValue }), type: 'addNode' },
    type: 'applyWorkflowAction',
  });
  state = workbenchReducer(state, {
    action: {
      edge: edge('gen', 'strings', 'batch', 'strings'),
      node: invocationNode('batch', 'string_batch', { batch_group_id: 'None' }),
      type: 'addNodeAndEdge',
    },
    type: 'applyWorkflowAction',
  });
  state = workbenchReducer(state, {
    action: {
      edge: edge('batch', 'value', 'prompt', 'value'),
      node: invocationNode('prompt', 'string', { value: 'static' }),
      type: 'addNodeAndEdge',
    },
    type: 'applyWorkflowAction',
  });

  return getActiveProject(state);
};
const workflowRoute: InvocationRoute = {
  destination: 'gallery',
  destinationLocked: false,
  sourceId: 'workflow',
  sourceLocked: false,
};

describe('submitResolvedInvocation with a workflow generator', () => {
  beforeEach(() => {
    generatorsMock.resolveWorkflowGenerators.mockReset();
  });

  const submit = async () => {
    const project = buildProject();
    const route = resolveInvocationRoute(project, 'global', workflowRoute);
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const notify = vi.spyOn(commands.notifications, 'add');

    expect(route.validationReasons).toEqual([]);
    expect(route.workflowBatch).toEqual({ size: null });

    await submitResolvedInvocation({
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

    return { notify, submitResolved };
  };

  it('reads the board listing alongside generator resolution and dispatches only once both have answered', async () => {
    galleryQueriesMock.ensureGalleryBoardDestinations.mockClear();
    let settleBoards!: () => void;
    galleryQueriesMock.ensureGalleryBoardDestinations.mockReturnValue(
      new Promise<void>((resolve) => {
        settleBoards = resolve;
      })
    );
    let settleGenerators!: () => void;
    const generators = new Promise<void>((resolve) => {
      settleGenerators = resolve;
    });
    generatorsMock.resolveWorkflowGenerators.mockImplementation(
      async (_client: unknown, pending: Array<{ key: string; nodeId: string }>) => {
        await generators;
        return {
          errors: [],
          resolutions: Object.fromEntries(pending.map((item) => [item.nodeId, { items: ['a cat'], key: item.key }])),
        };
      }
    );
    const project = buildProject();
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const settle = () =>
      new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });

    try {
      const submission = submitResolvedInvocation({
        commands,
        formatControlLayerError,
        formatUpscaleMetadataFailure,
        scopedDefaults: {},
        models: undefined,
        owner: captureAccountScope(),
        prepareCanvasInvocation: vi.fn(),
        project,
        route: resolveInvocationRoute(project, 'global', workflowRoute),
      });

      // The listing read did not hold generator resolution back: both started at once.
      expect(galleryQueriesMock.ensureGalleryBoardDestinations).toHaveBeenCalledTimes(1);
      await vi.waitFor(() => expect(generatorsMock.resolveWorkflowGenerators).toHaveBeenCalledTimes(1));

      // Generators answered; the dispatch still waits for the listing the destination is judged from.
      settleGenerators();
      await settle();
      expect(submitResolved).not.toHaveBeenCalled();

      settleBoards();
      await submission;
      expect(submitResolved).toHaveBeenCalledTimes(1);
    } finally {
      galleryQueriesMock.ensureGalleryBoardDestinations.mockReturnValue(null);
    }
  });

  it('resolves the pending generator and dispatches its output with the snapshot', async () => {
    generatorsMock.resolveWorkflowGenerators.mockImplementation(
      (_client: unknown, pending: Array<{ key: string; nodeId: string }>) =>
        Promise.resolve({
          errors: [],
          resolutions: Object.fromEntries(
            pending.map((item) => [item.nodeId, { items: ['a red cat', 'a green cat'], key: item.key }])
          ),
        })
    );

    const { notify, submitResolved } = await submit();

    expect(generatorsMock.resolveWorkflowGenerators).toHaveBeenCalledTimes(1);
    expect(generatorsMock.resolveWorkflowGenerators.mock.calls[0]?.[1]).toMatchObject([
      {
        nodeId: 'gen',
        request: { combinatorial: true, kind: 'dynamicPrompts', maxPrompts: 4, prompt: 'a {red|green} cat' },
      },
    ]);
    expect(notify).not.toHaveBeenCalled();
    expect(submitResolved).toHaveBeenCalledTimes(1);
    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject({
      workflowGenerators: { gen: { items: ['a red cat', 'a green cat'] } },
    });
  });

  it('carries the captured project and workflow through generator resolution to the dispatch', async () => {
    generatorsMock.resolveWorkflowGenerators.mockImplementation(
      (_client: unknown, pending: Array<{ key: string; nodeId: string }>) =>
        Promise.resolve({
          errors: [],
          resolutions: Object.fromEntries(pending.map((item) => [item.nodeId, { items: ['a cat'], key: item.key }])),
        })
    );
    const project = buildProject();
    const workflowId = project.workflows.activeWorkflowId;
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');

    await submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      project,
      route: resolveInvocationRoute(project, 'global', workflowRoute),
      workflowId,
    });

    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject({ projectId: project.id, workflowId });
  });

  it('queues nothing for a workflow the captured project no longer owns', async () => {
    const project = buildProject();
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');

    await submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      project,
      route: resolveInvocationRoute(project, 'global', workflowRoute),
      workflowId: 'removed',
    });

    expect(generatorsMock.resolveWorkflowGenerators).not.toHaveBeenCalled();
    expect(submitResolved).not.toHaveBeenCalled();
  });

  it('reports a generator that failed to resolve and queues nothing', async () => {
    generatorsMock.resolveWorkflowGenerators.mockResolvedValue({
      errors: ['string_generator: Unbalanced braces'],
      resolutions: {},
    });

    const { notify, submitResolved } = await submit();

    expect(submitResolved).not.toHaveBeenCalled();
    expect(notify).toHaveBeenCalledWith({
      kind: 'error',
      message: 'string_generator: Unbalanced braces',
      title: 'The batch generator could not be resolved',
      titleKey: 'topbar.invoke.generatorFailed',
    });
  });

  it('reports a batch that only turns out empty once its generator resolved', async () => {
    generatorsMock.resolveWorkflowGenerators.mockImplementation(
      (_client: unknown, pending: Array<{ key: string; nodeId: string }>) =>
        Promise.resolve({
          errors: [],
          resolutions: Object.fromEntries(pending.map((item) => [item.nodeId, { items: [], key: item.key }])),
        })
    );

    const { notify, submitResolved } = await submit();

    expect(submitResolved).not.toHaveBeenCalled();
    expect(notify).toHaveBeenCalledWith({
      kind: 'error',
      message: 'Batch node "string_batch" has an empty collection.',
      messageKey: 'nodes.readiness.batchEmpty',
      messageValues: { node: 'string_batch' },
      title: 'The batch is not ready',
      titleKey: 'topbar.invoke.batchNotReady',
    });
  });
});

describe('submitResolvedInvocation with a random dynamic prompt', () => {
  // Seed capabilities to match app boot; submission fails closed without them.
  seedArchitectureCapabilities();

  const drawn = { positivePrompts: ['a green cat'], positivePromptsSampleSeed: 7 };

  const submitDrawn = async (sourceId: 'canvas' | 'generate', prepend?: boolean) => {
    const model: MainModelConfig = { base: 'sd-1', key: 'main', name: 'Main', type: 'main' };
    const initial = createInitialWorkbenchState();
    const state = workbenchReducer(
      sourceId === 'canvas' ? workbenchReducer(initial, { presetId: 'edit', type: 'applyPreset' }) : initial,
      {
        type: 'setGenerateSettings',
        values: {
          ...getDefaultGenerateSettings(model),
          dynamicPromptsCombinatorial: false,
          dynamicPromptsMaxPrompts: 4,
          dynamicPromptsSampleSeed: 7,
          model,
          modelKey: model.key,
          positivePrompt: 'a {red|green} cat',
        },
      }
    );
    const project = getActiveProject(state);
    // Infinite staleTime makes the cached draw authoritative, so no request leaves the test.
    queryClient.setQueryData(
      dynamicPromptsKeys.expansion({ combinatorial: false, max_prompts: 4, prompt: 'a {red|green} cat', seed: 7 }),
      { error: null, prompts: ['a green cat'] }
    );
    const route = resolveInvocationRoute(project, 'global', {
      destination: 'gallery',
      destinationLocked: false,
      sourceId,
      sourceLocked: false,
    });
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn();

    expect(route.validationReasons).toEqual([]);

    await submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation,
      ...(prepend === undefined ? {} : { prepend }),
      project,
      route,
    });

    return { prepareCanvasInvocation, submitResolved };
  };

  it('dispatches Generate with the sample seed the expansion was drawn with', async () => {
    const { submitResolved } = await submitDrawn('generate');

    expect(submitResolved).toHaveBeenCalledTimes(1);
    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject(drawn);
  });

  it('hands Canvas the expansion and the sample seed it was drawn with', async () => {
    const { prepareCanvasInvocation, submitResolved } = await submitDrawn('canvas');

    expect(submitResolved).not.toHaveBeenCalled();
    expect(prepareCanvasInvocation).toHaveBeenCalledWith(expect.objectContaining({ expansion: drawn }));
  });

  it('keeps Invoke Front through prompt expansion to the Generate dispatch and the Canvas preparation', async () => {
    const generate = await submitDrawn('generate', true);
    const canvas = await submitDrawn('canvas', true);

    expect(generate.submitResolved.mock.calls[0]?.[0]).toMatchObject({ prepend: true });
    expect(canvas.prepareCanvasInvocation).toHaveBeenCalledWith(expect.objectContaining({ prepend: true }));
  });
});

describe('submitResolvedInvocation without dynamic prompts', () => {
  seedArchitectureCapabilities();

  const submitGenerate = async (prepend?: boolean, positivePrompt = 'a cat') => {
    const model: MainModelConfig = { base: 'sd-1', key: 'main', name: 'Main', type: 'main' };
    const project = getActiveProject(
      workbenchReducer(createInitialWorkbenchState(), {
        type: 'setGenerateSettings',
        values: { ...getDefaultGenerateSettings(model), model, modelKey: model.key, positivePrompt },
      })
    );
    const route = resolveInvocationRoute(project, 'global', {
      destination: 'gallery',
      destinationLocked: false,
      sourceId: 'generate',
      sourceLocked: false,
    });
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');

    expect(route.validationReasons).toEqual([]);

    await submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      ...(prepend === undefined ? {} : { prepend }),
      project,
      route,
    });

    return submitResolved;
  };

  it('dispatches a plain Generate prompt to the front of the queue only for Invoke Front', async () => {
    const front = await submitGenerate(true);
    const back = await submitGenerate();

    expect(front).toHaveBeenCalledTimes(1);
    expect(front.mock.calls[0]?.[0]).toMatchObject({ prepend: true });
    // No expansion took this path.
    expect(front.mock.calls[0]?.[0]).not.toHaveProperty('positivePrompts');
    expect(back.mock.calls[0]?.[0]).not.toHaveProperty('prepend');
  });

  it('dispatches a JSON prompt literally, without expanding its braces', async () => {
    const draft =
      '```json\n{"width":1024,"height":1536,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"caption":"a poster"},"items":[{"label":"object","caption":"a cup","bbox":[0,0,10,10]}]}\n```';
    const compiled =
      '{"bbox_coord_space":"abs_centralized","bbox_format":"yxyx","summary":{"caption":"a poster"},"items":[]}';
    for (const prompt of [draft, compiled]) {
      const submitted = await submitGenerate(undefined, prompt);
      expect(submitted).toHaveBeenCalledTimes(1);
      expect(submitted.mock.calls[0]?.[0]).not.toHaveProperty('positivePrompts');
      submitted.mockRestore();
    }
  });

  it('keeps Invoke Front on a Workflow dispatch', async () => {
    generatorsMock.resolveWorkflowGenerators.mockImplementation(
      (_client: unknown, pending: Array<{ key: string; nodeId: string }>) =>
        Promise.resolve({
          errors: [],
          resolutions: Object.fromEntries(pending.map((item) => [item.nodeId, { items: ['a cat'], key: item.key }])),
        })
    );
    const project = buildProject();
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');

    await submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      prepend: true,
      project,
      route: resolveInvocationRoute(project, 'global', workflowRoute),
    });

    expect(submitResolved).toHaveBeenCalledTimes(1);
    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject({ prepend: true });
  });
});

describe("submitResolvedInvocation reads the user's defaults", () => {
  seedArchitectureCapabilities();

  it('hands Canvas the compositing and generation flags the defaults give a project that pins none', async () => {
    const model: MainModelConfig = { base: 'sd-1', key: 'main', name: 'Main', type: 'main' };
    const state = workbenchReducer(
      workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' }),
      {
        type: 'setGenerateSettings',
        values: { ...getDefaultGenerateSettings(model), model, modelKey: model.key, positivePrompt: 'a cat' },
      }
    );
    const project = getActiveProject(state);
    const route = resolveInvocationRoute(project, 'global', {
      destination: 'gallery',
      destinationLocked: false,
      sourceId: 'canvas',
      sourceLocked: false,
    });
    const prepareCanvasInvocation = vi.fn();

    expect(route.validationReasons).toEqual([]);

    await submitResolvedInvocation({
      commands: createWorkbenchStore().commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {
        canvas: { outputOnlyMaskedRegions: false },
        generate: { useCpuNoise: false, useNsfwChecker: true, useWatermark: true },
      },
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation,
      project,
      route,
    });

    expect(prepareCanvasInvocation).toHaveBeenCalledWith(
      expect.objectContaining({
        compositing: expect.objectContaining({ outputOnlyMaskedRegions: false }),
        projectSettings: expect.objectContaining({ useCpuNoise: false, useNsfwChecker: true, useWatermark: true }),
      })
    );
  });
});

describe('submitResolvedInvocation for Upscale', () => {
  const upscaler = {
    base: 'any',
    file_size: 1,
    format: 'checkpoint',
    hash: 'ultrasharp-hash',
    key: 'ultrasharp',
    name: '4x-UltraSharp',
    path: 'ultrasharp',
    scale: 4,
    source: 'ultrasharp',
    source_type: 'path',
    type: 'spandrel_image_to_image',
  } as const;
  const upscaleRoute: InvocationRoute = {
    destination: 'gallery',
    destinationLocked: false,
    sourceId: 'upscale',
    sourceLocked: false,
  };

  beforeEach(() => {
    galleryMock.metadata.mockReset();
  });

  const submit = async (mode: 'refine' | 'model-only') => {
    const state = workbenchReducer(createInitialWorkbenchState(), {
      type: 'patchWidgetValues',
      values: {
        ...createDefaultUpscaleWidgetValues([upscaler]),
        inputImage: { height: 64, image_name: 'input.png', width: 64 },
        mode,
      },
      widgetId: 'upscale',
    });
    const project = getActiveProject(state);
    const commands = createWorkbenchStore().commands;
    const submitResolved = vi.spyOn(commands.generation, 'submitResolved');
    const notify = vi.spyOn(commands.notifications, 'add');

    await submitResolvedInvocation({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      scopedDefaults: {},
      models: [upscaler],
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      project,
      route: resolveInvocationRoute(project, 'global', upscaleRoute, [upscaler]),
    });

    return { notify, submitResolved };
  };

  it("reads a Model only input's metadata once and dispatches it with the image it was read for", async () => {
    galleryMock.metadata.mockResolvedValue({ positive_prompt: 'a lighthouse' });

    const { notify, submitResolved } = await submit('model-only');

    expect(galleryMock.metadata).toHaveBeenCalledTimes(1);
    expect(galleryMock.metadata.mock.calls[0]?.[0]).toBe('input.png');
    expect(notify).not.toHaveBeenCalled();
    expect(submitResolved).toHaveBeenCalledTimes(1);
    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject({
      upscaleSourceMetadata: { imageName: 'input.png', metadata: { positive_prompt: 'a lighthouse' } },
    });
  });

  it('dispatches an image the server has no metadata for with none to copy', async () => {
    galleryMock.metadata.mockRejectedValue(new ApiError('{"detail":"Image not found"}', 404));

    const { submitResolved } = await submit('model-only');

    expect(submitResolved.mock.calls[0]?.[0]).toMatchObject({
      upscaleSourceMetadata: { imageName: 'input.png', metadata: null },
    });
  });

  it('queues nothing and names the image when the metadata read fails', async () => {
    galleryMock.metadata.mockRejectedValue(new ApiError('{"detail":"Database is locked"}', 500));

    const { notify, submitResolved } = await submit('model-only');

    expect(submitResolved).not.toHaveBeenCalled();
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'error', message: 'Database is locked', title: 'metadataFailed: input.png' })
    );
  });

  it('reads no metadata for Refine, which records its own', async () => {
    const { submitResolved } = await submit('refine');

    expect(galleryMock.metadata).not.toHaveBeenCalled();
    expect(submitResolved.mock.calls[0]?.[0]).not.toHaveProperty('upscaleSourceMetadata');
  });
});

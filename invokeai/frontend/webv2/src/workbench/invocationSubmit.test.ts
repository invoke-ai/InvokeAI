import type { InvocationRoute } from '@workbench/invocationContracts';

import { captureAccountScope } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const templatesMock = vi.hoisted(() => ({
  snapshot: { error: null, status: 'idle', templates: {} } as {
    error: string | null;
    status: 'idle' | 'loading' | 'loaded' | 'error';
    templates: Record<string, unknown>;
  },
}));
const generatorsMock = vi.hoisted(() => ({ resolveWorkflowGenerators: vi.fn() }));

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
    type: 'applyProjectGraphAction',
  });
  state = workbenchReducer(state, {
    action: {
      edge: edge('gen', 'strings', 'batch', 'strings'),
      node: invocationNode('batch', 'string_batch', { batch_group_id: 'None' }),
      type: 'addNodeAndEdge',
    },
    type: 'applyProjectGraphAction',
  });
  state = workbenchReducer(state, {
    action: {
      edge: edge('batch', 'value', 'prompt', 'value'),
      node: invocationNode('prompt', 'string', { value: 'static' }),
      type: 'addNodeAndEdge',
    },
    type: 'applyProjectGraphAction',
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
      models: undefined,
      owner: captureAccountScope(),
      prepareCanvasInvocation: vi.fn(),
      project,
      route,
    });

    return { notify, submitResolved };
  };

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
      title: 'The batch is not ready',
    });
  });
});

import type { MainModelConfig } from '@features/generation/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { describe, expect, it, vi } from 'vitest';

import type { WorkbenchCommands, WorkbenchQueries } from './workbenchStore';

import { submitActiveInvocation, type ActiveInvocationSubmissionRuntime } from './activeInvocationSubmission';
import { isInvocationPreparing } from './invocationPreparation';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

// A loaded definition for the one node the workflow route needs; readiness reads templates imperatively.
vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getInvocationTemplatesSnapshot: () => ({
    error: null,
    status: 'loaded',
    templates: {
      noop: { inputs: {}, nodePack: 'invokeai', outputs: {}, outputType: 'noop_output', title: 'Noop', type: 'noop' },
    },
  }),
}));

// Seed capabilities to match app boot; submission fails closed without them.
seedArchitectureCapabilities();

const owner = { signal: new AbortController().signal } as AccountScope;

const createArgs = (state = createInitialWorkbenchState()) => {
  return {
    commands: {} as WorkbenchCommands,
    formatControlLayerError: ({ code, layerName }: { code: string; layerName: string }) => `${layerName}: ${code}`,
    getModels: () => undefined,
    queries: {
      getSnapshot: () => ({
        account: state.account,
        activeProject: state.projects[0],
        autosave: state.autosave,
        backendConnection: state.backendConnection,
        hasHydrated: true,
        notifications: state.notifications,
        projects: state.projects,
      }),
    } as WorkbenchQueries,
  };
};

const model: MainModelConfig = { base: 'sd-1', key: 'main', name: 'Main', type: 'main' };

const withGenerateModel = (state: ReturnType<typeof createInitialWorkbenchState>) => ({
  ...workbenchReducer(state, {
    type: 'setGenerateSettings',
    values: { ...getDefaultGenerateSettings(model), model, modelKey: model.key },
  }),
  backendConnection: { status: 'connected' as const },
});

const createCanvasState = () =>
  withGenerateModel(workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' }));

const createGenerateState = () => withGenerateModel(createInitialWorkbenchState());

// The Automate preset mounts the Workflow widget; one node makes the graph runnable.
const createWorkflowState = () => {
  let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'automate', type: 'applyPreset' });
  state = workbenchReducer(state, {
    action: {
      node: {
        data: {
          inputs: {},
          isIntermediate: true,
          isOpen: true,
          label: '',
          nodePack: 'invokeai',
          notes: '',
          type: 'noop',
          useCache: true,
          version: '1.0.0',
        },
        id: 'noop-1',
        position: { x: 0, y: 0 },
        type: 'invocation',
      },
      type: 'addNode',
    },
    type: 'applyWorkflowAction',
  });
  state = workbenchReducer(state, { sourceId: 'workflow', type: 'setInvocationSource' });

  return { ...state, backendConnection: { status: 'connected' as const } };
};

describe('active invocation submission', () => {
  it('single-flights a canvas submission before its lazy module loads and until preparation settles', async () => {
    let resolveModule = (_value: { prepareCanvasInvocation: () => Promise<void> }): void => undefined;
    const modulePromise = new Promise<{ prepareCanvasInvocation: () => Promise<void> }>((resolve) => {
      resolveModule = resolve;
    });
    let resolvePreparation = (): void => undefined;
    const preparationPromise = new Promise<void>((resolve) => {
      resolvePreparation = resolve;
    });
    const events: string[] = [];
    const runtime: ActiveInvocationSubmissionRuntime = {
      assertCurrent: () => undefined,
      capture: () => owner,
      flushDrafts: () => events.push('flushed'),
      isCurrent: () => true,
      loadPrepareCanvasInvocation: () => {
        events.push('loaded');
        return modulePromise;
      },
      submit: () => {
        events.push('submitted');
        return preparationPromise;
      },
    };

    const state = createCanvasState();
    const projectId = state.activeProjectId;
    const first = submitActiveInvocation(createArgs(state), runtime);
    const second = submitActiveInvocation(createArgs(state), runtime);

    expect(events).toEqual(['flushed', 'loaded', 'flushed']);
    expect(isInvocationPreparing(projectId)).toBe(true);

    resolveModule({ prepareCanvasInvocation: () => preparationPromise });
    await Promise.resolve();
    await Promise.resolve();

    expect(events).toEqual(['flushed', 'loaded', 'flushed', 'submitted']);

    resolvePreparation();
    await Promise.all([first, second]);
    expect(isInvocationPreparing(projectId)).toBe(false);
  });

  // Generate awaits prompt expansion and workflow awaits generator resolution before dispatching.
  it.each([
    ['generate', createGenerateState],
    ['workflow', createWorkflowState],
  ] as const)('drops a %s submission made while an earlier one is still preparing', async (sourceId, createState) => {
    let resolvePreparation = (): void => undefined;
    const preparationPromise = new Promise<void>((resolve) => {
      resolvePreparation = resolve;
    });
    const submittedSources: string[] = [];
    const runtime: ActiveInvocationSubmissionRuntime = {
      assertCurrent: () => undefined,
      capture: () => owner,
      flushDrafts: () => undefined,
      isCurrent: () => true,
      loadPrepareCanvasInvocation: () => Promise.reject(new Error('Only Canvas loads the Canvas chunk')),
      submit: ({ route }) => {
        submittedSources.push(route.sourceId);
        return submittedSources.length === 1 ? preparationPromise : undefined;
      },
    };
    const state = createState();
    const projectId = state.activeProjectId;

    const first = submitActiveInvocation(createArgs(state), runtime);
    await Promise.resolve();
    const second = submitActiveInvocation(createArgs(state), runtime);
    await second;

    expect(submittedSources).toEqual([sourceId]);
    expect(isInvocationPreparing(projectId)).toBe(true);

    resolvePreparation();
    await first;
    expect(isInvocationPreparing(projectId)).toBe(false);

    // The released lease admits the next submission.
    await submitActiveInvocation(createArgs(state), runtime);
    expect(submittedSources).toEqual([sourceId, sourceId]);
  });

  it('swallows an asynchronous failure after the initiating account becomes stale', async () => {
    const events: string[] = [];
    const runtime: ActiveInvocationSubmissionRuntime = {
      assertCurrent: () => undefined,
      capture: () => owner,
      flushDrafts: () => events.push('flushed'),
      isCurrent: () => false,
      loadPrepareCanvasInvocation: () => Promise.reject(new Error('stale failure')),
      submit: () => {
        events.push('submitted');
      },
    };

    const state = createCanvasState();

    await expect(submitActiveInvocation(createArgs(state), runtime)).resolves.toBeUndefined();
    expect(events).toEqual(['flushed']);
    expect(isInvocationPreparing(state.activeProjectId)).toBe(false);
  });

  it('rethrows an asynchronous failure for the current account', async () => {
    const runtime: ActiveInvocationSubmissionRuntime = {
      assertCurrent: () => undefined,
      capture: () => owner,
      flushDrafts: () => undefined,
      isCurrent: () => true,
      loadPrepareCanvasInvocation: () => Promise.reject(new Error('current failure')),
      submit: () => undefined,
    };

    const state = createCanvasState();

    await expect(submitActiveInvocation(createArgs(state), runtime)).rejects.toThrow('current failure');
    expect(isInvocationPreparing(state.activeProjectId)).toBe(false);
  });
});

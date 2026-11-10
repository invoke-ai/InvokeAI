import type {
  ComponentModelConfig,
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { galleryKeys } from '@features/gallery/queries';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { queryClient } from '@platform/query/client';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { CanvasRegionalGuidanceLayerContract } from './canvas-engine/api';
import type { WorkbenchCommands, WorkbenchQueries } from './workbenchStore';

import { submitActiveInvocation, type ActiveInvocationSubmissionRuntime } from './activeInvocationSubmission';
import { beginInvocationPreparation, endInvocationPreparation, isInvocationPreparing } from './invocationPreparation';
import { submitResolvedInvocation } from './invocationSubmit';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';
import { createWorkbenchStore } from './workbenchStore';

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
    formatUpscaleMetadataFailure: (name: string) => name,
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

const withActiveSource = (state: ReturnType<typeof createInitialWorkbenchState>, sourceId: 'workflow') =>
  workbenchReducer(state, { sourceId, type: 'setInvocationSource' });

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
  it('forwards Invoke Front to the submission and omits it otherwise', async () => {
    const prepends: (boolean | undefined)[] = [];
    const runtime: ActiveInvocationSubmissionRuntime = {
      assertCurrent: () => undefined,
      capture: () => owner,
      flushDrafts: () => undefined,
      isCurrent: () => true,
      loadPrepareCanvasInvocation: () => Promise.reject(new Error('Only Canvas loads the Canvas chunk')),
      submit: (deps) => {
        prepends.push(deps.prepend);
      },
    };

    await submitActiveInvocation(createArgs(createGenerateState()), runtime);
    await submitActiveInvocation({ ...createArgs(createGenerateState()), prepend: true }, runtime);

    expect(prepends).toEqual([undefined, true]);
  });

  it('runs a given source’s route with that source fixed, telling the caller once it has started', async () => {
    const routes: { mode: string; sourceId: string; sourceLocked: boolean }[] = [];
    const runtime: ActiveInvocationSubmissionRuntime = {
      assertCurrent: () => undefined,
      capture: () => owner,
      flushDrafts: () => undefined,
      isCurrent: () => true,
      loadPrepareCanvasInvocation: () => Promise.reject(new Error('Only Canvas loads the Canvas chunk')),
      submit: ({ route }) => {
        routes.push({ mode: route.mode, sourceId: route.sourceId, sourceLocked: route.sourceLocked });
      },
    };
    const state = createGenerateState();
    const onStarted = vi.fn();

    // Workflow is the active source; the preview runs Generate's route.
    await submitActiveInvocation(
      { ...createArgs(withActiveSource(state, 'workflow')), onStarted, sourceId: 'generate' },
      runtime
    );

    expect(routes).toEqual([{ mode: 'dialog', sourceId: 'generate', sourceLocked: true }]);
    expect(onStarted).toHaveBeenCalledTimes(1);
  });

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

/**
 * A graph preview runs its own source's route (`sourceId`), resolved for real: these cases came from the preview's
 * former submission path and hold for this one.
 */
describe('active invocation submission for a graph preview’s source', () => {
  const formatControlLayerError = ({ code, layerName }: { code: string; layerName: string }) => `${layerName}: ${code}`;
  const animaModel: MainModelConfig = { base: 'anima', key: 'anima-model', name: 'Anima', type: 'main' };
  const animaVae: VaeModelConfig = { base: 'qwen-image', key: 'anima-vae', name: 'Anima VAE', type: 'vae' };
  const qwen3Encoder: ComponentModelConfig = {
    base: 'any',
    key: 'qwen3-encoder',
    name: 'Qwen3 Encoder',
    type: 'qwen3_encoder',
    variant: 'qwen3_06b',
  };

  const projectWith = (values: GenerateWidgetValues) => {
    const state = workbenchReducer(createInitialWorkbenchState(), { type: 'setGenerateSettings', values });

    return state.projects.find((candidate) => candidate.id === state.activeProjectId)!;
  };
  type TestProject = ReturnType<typeof projectWith>;
  // Mount Canvas, so the route counts that source available.
  const withCanvasMounted = (project: TestProject): TestProject => ({
    ...project,
    widgetRegions: {
      ...project.widgetRegions,
      center: { ...project.widgetRegions.center, activeInstanceId: 'canvas', instanceIds: ['canvas'] },
    },
  });
  const animaProject = () =>
    projectWith({
      ...getDefaultGenerateSettings(animaModel),
      model: animaModel,
      modelKey: animaModel.key,
      positivePrompt: 'an anima prompt',
      qwen3EncoderModel: qwen3Encoder,
      vae: animaVae,
    });

  /** The real submission with the real route, Canvas preparation recorded rather than run. */
  const submitPreview = async (project: TestProject, sourceId: 'canvas' | 'generate') => {
    const store = createWorkbenchStore();
    const submitResolved = vi.spyOn(store.commands.generation, 'submitResolved');
    const prepareCanvasInvocation = vi.fn(() => Promise.resolve());
    const onStarted = vi.fn();
    const runtime: ActiveInvocationSubmissionRuntime = {
      assertCurrent: () => undefined,
      // The current account, which the submission checks before it dispatches.
      capture: captureAccountScope,
      flushDrafts: () => undefined,
      isCurrent: () => true,
      loadPrepareCanvasInvocation: () => Promise.resolve({ prepareCanvasInvocation }),
      submit: submitResolvedInvocation,
    };

    await submitActiveInvocation(
      {
        commands: store.commands,
        formatControlLayerError,
        formatUpscaleMetadataFailure: (name: string) => name,
        getModels: () => undefined,
        onStarted,
        queries: {
          getSnapshot: () => ({
            activeProject: project,
            backendConnection: { status: 'connected' },
            scopedDefaults: {},
          }),
        } as unknown as WorkbenchQueries,
        sourceId,
      },
      runtime
    );

    return {
      prepared: prepareCanvasInvocation.mock.calls as unknown as [{ projectId: string }][],
      started: onStarted.mock.calls.length,
      submitted: submitResolved.mock.calls.length,
    };
  };

  beforeEach(() => {
    queryClient.clear();
    // A fresh listing of the readable boards, as the destinations check leaves it: none named.
    queryClient.setQueryData(
      galleryKeys.boards(captureAccountScope(), {
        includeArchived: true,
        includeDateBoards: false,
        orderBy: 'created_at',
        orderDir: 'DESC',
      }),
      []
    );
  });

  it('runs a Canvas source through Canvas preparation, not a resolved snapshot', async () => {
    const project = withCanvasMounted(animaProject());
    const { prepared, started, submitted } = await submitPreview(project, 'canvas');

    expect(started).toBe(1);
    expect(submitted).toBe(0);
    expect(prepared).toHaveLength(1);
    expect(prepared[0]?.[0]).toMatchObject({ projectId: project.id });
  });

  it('prepares nothing for a Canvas preview of a blank prompt around a layout region while Generate is active', async () => {
    const ideogramModel: MainModelConfig = {
      base: 'ideogram-4',
      format: 'diffusers',
      key: 'ideogram-4-model',
      name: 'Ideogram 4',
      type: 'main',
    };
    const region: CanvasRegionalGuidanceLayerContract = {
      autoNegative: false,
      blendMode: 'normal',
      id: 'region',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: { height: 64, imageName: 'region.png', width: 64 }, fill: { color: '#00ff00', style: 'solid' } },
      name: 'Region',
      negativePrompt: null,
      opacity: 1,
      positivePrompt: 'a red kite',
      referenceImages: [],
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 32, y: 32 },
      type: 'regional_guidance',
    };
    const preview = async (positivePrompt: string) => {
      const project = withCanvasMounted(
        projectWith({
          ...getDefaultGenerateSettings(ideogramModel),
          model: ideogramModel,
          modelKey: ideogramModel.key,
          positivePrompt,
        })
      );
      const { prepared, started } = await submitPreview(
        {
          ...project,
          canvas: {
            ...project.canvas,
            document: {
              ...project.canvas.document,
              bbox: { height: 512, width: 512, x: 0, y: 0 },
              stacks: { ...project.canvas.document.stacks, regional_guidance: [region] },
            },
          },
          invocation: { ...project.invocation, sourceId: 'generate' },
        },
        'canvas'
      );

      return { prepared: prepared.length, started };
    };

    // Nothing uploads for a submission the compiler would reject; with a prompt the same preview goes ahead.
    expect(await preview('')).toEqual({ prepared: 0, started: 0 });
    expect(await preview('a beach at noon')).toEqual({ prepared: 1, started: 1 });
  });

  it('starts nothing while another submission for the project holds the lease', async () => {
    const project = animaProject();
    const lease = beginInvocationPreparation(project.id);

    try {
      expect(await submitPreview(project, 'generate')).toEqual({ prepared: [], started: 0, submitted: 0 });
    } finally {
      endInvocationPreparation(lease!);
    }

    expect(await submitPreview(project, 'generate')).toMatchObject({ started: 1, submitted: 1 });
  });

  it('starts nothing for a route that cannot run (a source the layout does not hold)', async () => {
    expect(await submitPreview(animaProject(), 'canvas')).toEqual({ prepared: [], started: 0, submitted: 0 });
  });
});

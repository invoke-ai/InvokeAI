import type { MainModelConfig } from '@features/generation/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { describe, expect, it } from 'vitest';

import type { WorkbenchCommands, WorkbenchQueries } from './workbenchStore';

import { submitActiveInvocation, type ActiveInvocationSubmissionRuntime } from './activeInvocationSubmission';
import { isCanvasInvocationPreparing } from './canvasInvocationPreparation';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

const owner = { signal: new AbortController().signal } as AccountScope;

const createArgs = (state = createInitialWorkbenchState()) => {
  return {
    commands: {} as WorkbenchCommands,
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

const createCanvasState = () => {
  const model: MainModelConfig = { base: 'sd-1', key: 'main', name: 'Main', type: 'main' };
  let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
  state = workbenchReducer(state, {
    type: 'setGenerateSettings',
    values: { ...getDefaultGenerateSettings(model), model, modelKey: model.key },
  });

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
    expect(isCanvasInvocationPreparing(projectId)).toBe(true);

    resolveModule({ prepareCanvasInvocation: () => preparationPromise });
    await Promise.resolve();
    await Promise.resolve();

    expect(events).toEqual(['flushed', 'loaded', 'flushed', 'submitted']);

    resolvePreparation();
    await Promise.all([first, second]);
    expect(isCanvasInvocationPreparing(projectId)).toBe(false);
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

    await expect(submitActiveInvocation(createArgs(createCanvasState()), runtime)).resolves.toBeUndefined();
    expect(events).toEqual(['flushed']);
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

    await expect(submitActiveInvocation(createArgs(createCanvasState()), runtime)).rejects.toThrow('current failure');
  });
});

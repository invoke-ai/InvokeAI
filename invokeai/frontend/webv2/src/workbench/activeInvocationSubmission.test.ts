import type { AccountScope } from '@platform/state/accountLifecycle';

import { describe, expect, it } from 'vitest';

import type { WorkbenchCommands, WorkbenchQueries } from './workbenchStore';

import { submitActiveInvocation, type ActiveInvocationSubmissionRuntime } from './activeInvocationSubmission';
import { createInitialWorkbenchState } from './workbenchState.testing';

const owner = { signal: new AbortController().signal } as AccountScope;

const createArgs = () => {
  const state = createInitialWorkbenchState();

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

describe('active invocation submission', () => {
  it('swallows an asynchronous failure after the initiating account becomes stale', async () => {
    const events: string[] = [];
    const runtime: ActiveInvocationSubmissionRuntime = {
      assertCurrent: () => undefined,
      capture: () => owner,
      flushDrafts: () => events.push('flushed'),
      isCurrent: () => false,
      loadPrepareCanvasInvocation: () => Promise.reject(new Error('stale failure')),
      submit: () => events.push('submitted'),
    };

    await expect(submitActiveInvocation(createArgs(), runtime)).resolves.toBeUndefined();
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

    await expect(submitActiveInvocation(createArgs(), runtime)).rejects.toThrow('current failure');
  });
});

import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { flushGenerateDrafts } from '@features/generation/react';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

import type { ResultDestination } from './invocationContracts';
import type { SubmitResolvedInvocationDeps } from './invocationSubmit';
import type { PrepareCanvasInvocationArgs } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands, WorkbenchQueries } from './workbenchStore';

import { isInvocationRouteValid, resolveInvocationRoute } from './invocation';
import { submitResolvedInvocation } from './invocationSubmit';

export interface ActiveInvocationSubmissionArgs {
  commands: WorkbenchCommands;
  destinationOverride?: ResultDestination;
  formatControlLayerError?: PrepareCanvasInvocationArgs['formatControlLayerError'];
  getModels: () => readonly ModelConfig[] | undefined;
  queries: WorkbenchQueries;
}

export interface ActiveInvocationSubmissionRuntime {
  assertCurrent: (owner: AccountScope) => void;
  capture: () => AccountScope;
  flushDrafts: () => void;
  isCurrent: (owner: AccountScope) => boolean;
  loadPrepareCanvasInvocation: () => Promise<{
    prepareCanvasInvocation: SubmitResolvedInvocationDeps['prepareCanvasInvocation'];
  }>;
  submit: (args: SubmitResolvedInvocationDeps) => void;
}

const runtime: ActiveInvocationSubmissionRuntime = {
  assertCurrent: assertAccountScopeCurrent,
  capture: captureAccountScope,
  flushDrafts: flushGenerateDrafts,
  isCurrent: isAccountScopeCurrent,
  loadPrepareCanvasInvocation: () => import('./widgets/canvas/invoke/prepareCanvasInvocation'),
  submit: submitResolvedInvocation,
};

export const submitActiveInvocation = async (
  { commands, destinationOverride, formatControlLayerError, getModels, queries }: ActiveInvocationSubmissionArgs,
  activeRuntime: ActiveInvocationSubmissionRuntime = runtime
): Promise<void> => {
  const owner = activeRuntime.capture();
  activeRuntime.flushDrafts();

  try {
    const { prepareCanvasInvocation } = await activeRuntime.loadPrepareCanvasInvocation();

    activeRuntime.assertCurrent(owner);
    const snapshot = queries.getSnapshot();
    const invocation = destinationOverride
      ? { ...snapshot.activeProject.invocation, destination: destinationOverride }
      : snapshot.activeProject.invocation;
    const models = getModels();
    const route = resolveInvocationRoute(snapshot.activeProject, 'global', invocation, models);

    if (!isInvocationRouteValid(route) || snapshot.backendConnection.status !== 'connected') {
      return;
    }

    activeRuntime.submit({
      commands,
      formatControlLayerError,
      models,
      owner,
      prepareCanvasInvocation,
      project: snapshot.activeProject,
      route,
    });
  } catch (error) {
    if (!activeRuntime.isCurrent(owner)) {
      return;
    }

    throw error;
  }
};

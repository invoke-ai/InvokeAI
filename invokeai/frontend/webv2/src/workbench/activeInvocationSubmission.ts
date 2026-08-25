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

import { beginCanvasInvocationPreparation, endCanvasInvocationPreparation } from './canvasInvocationPreparation';
import { createDeferredResource } from './deferredResource';
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
  submit: (args: SubmitResolvedInvocationDeps) => Promise<void> | void;
}

type CanvasInvocationModule = {
  prepareCanvasInvocation: SubmitResolvedInvocationDeps['prepareCanvasInvocation'];
};

const canvasInvocationModule = createDeferredResource<CanvasInvocationModule>(
  () => import('./widgets/canvas/invoke/prepareCanvasInvocation')
);

const loadCanvasInvocationModule = (): Promise<CanvasInvocationModule> =>
  canvasInvocationModule.getStatus() === 'failed' ? canvasInvocationModule.retry() : canvasInvocationModule.load();

/** Warms the canvas-only invocation chunk without adding it to the eager workbench bundle. */
export const preloadCanvasInvocation = (): void => canvasInvocationModule.preload();

const runtime: ActiveInvocationSubmissionRuntime = {
  assertCurrent: assertAccountScopeCurrent,
  capture: captureAccountScope,
  flushDrafts: flushGenerateDrafts,
  isCurrent: isAccountScopeCurrent,
  loadPrepareCanvasInvocation: loadCanvasInvocationModule,
  submit: submitResolvedInvocation,
};

export const submitActiveInvocation = async (
  { commands, destinationOverride, formatControlLayerError, getModels, queries }: ActiveInvocationSubmissionArgs,
  activeRuntime: ActiveInvocationSubmissionRuntime = runtime
): Promise<void> => {
  const owner = activeRuntime.capture();
  activeRuntime.flushDrafts();
  const snapshot = queries.getSnapshot();
  const invocation = destinationOverride
    ? { ...snapshot.activeProject.invocation, destination: destinationOverride }
    : snapshot.activeProject.invocation;
  const models = getModels();
  const route = resolveInvocationRoute(snapshot.activeProject, 'global', invocation, models);

  if (!isInvocationRouteValid(route) || snapshot.backendConnection.status !== 'connected') {
    return;
  }

  const projectId = snapshot.activeProject.id;
  const isCanvasSubmission = route.sourceId === 'canvas';
  const preparationLease = isCanvasSubmission ? beginCanvasInvocationPreparation(projectId) : null;

  if (isCanvasSubmission && !preparationLease) {
    return;
  }

  try {
    const prepareCanvasInvocation = isCanvasSubmission
      ? (await activeRuntime.loadPrepareCanvasInvocation()).prepareCanvasInvocation
      : () => Promise.resolve();

    activeRuntime.assertCurrent(owner);
    await activeRuntime.submit({
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
  } finally {
    if (preparationLease) {
      endCanvasInvocationPreparation(preparationLease);
    }
  }
};

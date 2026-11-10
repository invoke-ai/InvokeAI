import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { flushGenerateDrafts } from '@features/generation/react';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

import type { InvocationSourceId, ResultDestination } from './invocationContracts';
import type { SubmitResolvedInvocationDeps } from './invocationSubmit';
import type { PrepareCanvasInvocationArgs } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands, WorkbenchQueries } from './workbenchStore';

import { createDeferredResource } from './deferredResource';
import { isInvocationRouteValid, resolveInvocationRoute } from './invocation';
import { beginInvocationPreparation, endInvocationPreparation } from './invocationPreparation';
import { submitResolvedInvocation } from './invocationSubmit';

export interface ActiveInvocationSubmissionArgs {
  commands: WorkbenchCommands;
  destinationOverride?: ResultDestination;
  /** Words a control-layer rejection from the locale; its error message is for logs only. */
  formatControlLayerError: PrepareCanvasInvocationArgs['formatControlLayerError'];
  formatUpscaleMetadataFailure: SubmitResolvedInvocationDeps['formatUpscaleMetadataFailure'];
  getModels: () => readonly ModelConfig[] | undefined;
  /** Invoke Front: the backend runs this batch ahead of the user's other pending work. */
  prepend?: boolean;
  queries: WorkbenchQueries;
  /**
   * Runs this source's route, its source fixed, as a graph preview of that source does, rather than the active route.
   * The destination stays the active route's.
   */
  sourceId?: InvocationSourceId;
  /** Hears that the submission has started: it is valid and holds the project's preparation lease. */
  onStarted?: () => void;
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
  {
    commands,
    destinationOverride,
    formatControlLayerError,
    formatUpscaleMetadataFailure,
    getModels,
    prepend,
    queries,
    sourceId,
    onStarted,
  }: ActiveInvocationSubmissionArgs,
  activeRuntime: ActiveInvocationSubmissionRuntime = runtime
): Promise<void> => {
  const owner = activeRuntime.capture();
  activeRuntime.flushDrafts();
  const snapshot = queries.getSnapshot();
  const active = snapshot.activeProject.invocation;
  const invocation = {
    ...active,
    ...(destinationOverride ? { destination: destinationOverride } : {}),
    ...(sourceId ? { sourceId, sourceLocked: true } : {}),
  };
  const models = getModels();
  const route = resolveInvocationRoute(snapshot.activeProject, sourceId ? 'dialog' : 'global', invocation, models);

  if (!isInvocationRouteValid(route) || snapshot.backendConnection.status !== 'connected') {
    return;
  }

  const isCanvasSubmission = route.sourceId === 'canvas';
  // Every route may await (Canvas chunk, prompt expansion, workflow generators) on settings captured above.
  const preparationLease = beginInvocationPreparation(snapshot.activeProject.id);

  if (!preparationLease) {
    return;
  }

  onStarted?.();

  try {
    const prepareCanvasInvocation = isCanvasSubmission
      ? (await activeRuntime.loadPrepareCanvasInvocation()).prepareCanvasInvocation
      : () => Promise.resolve();

    activeRuntime.assertCurrent(owner);
    await activeRuntime.submit({
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      models,
      owner,
      prepareCanvasInvocation,
      ...(prepend ? { prepend } : {}),
      project: snapshot.activeProject,
      route,
      scopedDefaults: snapshot.scopedDefaults,
    });
  } catch (error) {
    if (!activeRuntime.isCurrent(owner)) {
      return;
    }

    throw error;
  } finally {
    endInvocationPreparation(preparationLease);
  }
};

import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import type { InvocationSourceId } from './invocationContracts';
import type { Project } from './projectContracts';
import type { prepareCanvasInvocation } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands } from './workbenchStore';

import { isInvocationRouteValid, resolveInvocationRoute } from './invocation';
import { beginInvocationPreparation, endInvocationPreparation } from './invocationPreparation';
import { submitResolvedInvocation } from './invocationSubmit';

export interface GraphPreviewInvokeDeps {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  /** Words a blocked canvas preview's control layer for the notice. */
  formatControlLayerError: Parameters<typeof prepareCanvasInvocation>[0]['formatControlLayerError'];
  models: readonly ModelConfig[] | undefined;
  owner: AccountScope;
  prepareCanvasInvocation: typeof prepareCanvasInvocation;
  project: Project;
  sourceId: InvocationSourceId | undefined;
}

/**
 * Resolves and submits a preview against the post-draft-flush project snapshot. Shares the active submission's
 * preparation lease, so it reports false while another submission for the project is still preparing.
 */
export const resolveAndSubmitGraphPreviewInvocation = ({
  commands,
  formatControlLayerError,
  models,
  owner,
  prepareCanvasInvocation: prepareCanvas,
  project,
  sourceId,
}: GraphPreviewInvokeDeps): boolean => {
  if (!sourceId || !isAccountScopeCurrent(owner)) {
    return false;
  }

  const route = resolveInvocationRoute(
    project,
    'dialog',
    { ...project.invocation, sourceId, sourceLocked: true },
    models
  );

  if (!isInvocationRouteValid(route)) {
    return false;
  }

  const preparationLease = beginInvocationPreparation(project.id);

  if (!preparationLease) {
    return false;
  }

  void submitResolvedInvocation({
    commands,
    formatControlLayerError,
    models,
    owner,
    prepareCanvasInvocation: prepareCanvas,
    project,
    route,
  }).finally(() => endInvocationPreparation(preparationLease));
  return true;
};

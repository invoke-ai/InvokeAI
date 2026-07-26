import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import type { InvocationSourceId } from './invocationContracts';
import type { Project } from './projectContracts';
import type { prepareCanvasInvocation } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands } from './workbenchStore';

import { isInvocationRouteValid, resolveInvocationRoute } from './invocation';
import { submitResolvedInvocation } from './invocationSubmit';

export interface GraphPreviewInvokeDeps {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  models: readonly ModelConfig[] | undefined;
  owner: AccountScope;
  prepareCanvasInvocation: typeof prepareCanvasInvocation;
  project: Project;
  sourceId: InvocationSourceId | undefined;
}

/** Resolves and submits a preview against the post-draft-flush project snapshot. */
export const resolveAndSubmitGraphPreviewInvocation = ({
  commands,
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

  submitResolvedInvocation({ commands, models, owner, prepareCanvasInvocation: prepareCanvas, project, route });
  return true;
};

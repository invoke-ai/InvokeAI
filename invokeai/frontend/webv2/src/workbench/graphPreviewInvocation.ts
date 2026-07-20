import type { ModelConfig } from '@features/models';

import type { InvocationSourceId } from './invocationContracts';
import type { Project } from './projectContracts';
import type { prepareCanvasInvocation } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands } from './workbenchStore';

import { isInvocationRouteValid, resolveInvocationRoute } from './invocation';
import { submitResolvedInvocation } from './invocationSubmit';

export interface GraphPreviewInvokeDeps {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  models: readonly ModelConfig[] | undefined;
  prepareCanvasInvocation: typeof prepareCanvasInvocation;
  project: Project;
  sourceId: InvocationSourceId | undefined;
}

/** Resolves and submits a preview against the post-draft-flush project snapshot. */
export const resolveAndSubmitGraphPreviewInvocation = ({
  commands,
  models,
  prepareCanvasInvocation: prepareCanvas,
  project,
  sourceId,
}: GraphPreviewInvokeDeps): boolean => {
  if (!sourceId) {
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

  submitResolvedInvocation({ commands, models, prepareCanvasInvocation: prepareCanvas, project, route });
  return true;
};

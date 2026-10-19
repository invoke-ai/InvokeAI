import type { SocketHub } from '@platform/transport/socketHub';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { getProjectWidgetValues } from '@workbench/widgetState';

import type { PendingRecallEvent, RecallRevealContext, RecallRuntime } from './recallEventRuntime';

import { executeRecallParameters } from './executeRecallParameters';
import { bringRecallWidgetToFront, createRecallEventRuntime } from './recallEventRuntime';
import { isRecallParametersUpdatedEvent } from './recallParameters';

/** Apply `recall_parameters_updated` events to their arrival-time project's Generate panel. */
export const createRecallParametersRuntime = ({
  commands,
  getSessionUserId,
  hub,
  queries,
  replay,
  reveal,
  t,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications' | 'widgets'>;
  getSessionUserId?: () => string | null;
  hub: Pick<SocketHub, 'on'>;
  queries: Pick<WorkbenchQueries, 'getProject' | 'getSnapshot'>;
  replay?: readonly PendingRecallEvent[];
  reveal: RecallRevealContext;
  /** Resolves against the current language at call time; captured once, at attach. */
  t: TFunction;
}): RecallRuntime =>
  createRecallEventRuntime({
    apply: async ({ parameters }, { models, owner, projectId }) => {
      const applied = await executeRecallParameters({
        commands,
        t,
        getGenerateValues: () => {
          const project = queries.getProject(projectId);
          return project ? getProjectWidgetValues(project, 'generate') : null;
        },
        models,
        owner,
        parameters,
        projectId,
      });

      if (applied) {
        bringRecallWidgetToFront({ commands, owner, projectId, queries, reveal, typeId: 'generate' });
      }
    },
    area: 'recall-parameters',
    commands,
    eventName: 'recall_parameters_updated',
    getSessionUserId,
    hub,
    isEvent: isRecallParametersUpdatedEvent,
    queries,
    replay,
  });

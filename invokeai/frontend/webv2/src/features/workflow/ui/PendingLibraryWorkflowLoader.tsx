import { getLibraryWorkflow, touchLibraryWorkflowOpenedAt } from '@features/workflow/data/api';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import { clearPendingLibraryWorkflowLoad, workflowUiStore } from '@features/workflow/ui/workflowUiStore';
import { parseWorkflowJson } from '@features/workflow/utility';
import { getApiErrorMessage } from '@platform/transport/http';
import { useEffect, useEffectEvent } from 'react';

/**
 * Consumes `pendingLibraryWorkflowId` requests from shell surfaces (the
 * command palette) that cannot reach the graph context themselves: fetches
 * the library record, parses it, and replaces the project graph — the same
 * load path as WorkflowLibraryDialog, minus the dialog.
 */
export const PendingLibraryWorkflowLoader = () => {
  const { replace } = useProjectGraphCommands();
  const notify = useWorkflowNotifications();
  const pendingWorkflowId = workflowUiStore.useSelector((snapshot) => snapshot.pendingLibraryWorkflowId);

  const loadPendingWorkflow = useEffectEvent(async (workflowId: string) => {
    try {
      const raw = await getLibraryWorkflow(workflowId);
      const { document, warnings } = parseWorkflowJson(raw);
      const name = typeof raw.name === 'string' && raw.name.length > 0 ? raw.name : 'workflow';

      replace(document, `Loaded "${name}" from library`);

      for (const warning of warnings) {
        notify.info('Workflow load warning', warning);
      }

      void touchLibraryWorkflowOpenedAt(workflowId).catch(() => {
        // Recency bookkeeping only; loading already succeeded.
      });
    } catch (error) {
      notify.error('Failed to load workflow', getApiErrorMessage(error, 'Could not load the workflow.'));
    } finally {
      clearPendingLibraryWorkflowLoad();
    }
  });

  useEffect(() => {
    if (pendingWorkflowId) {
      void loadPendingWorkflow(pendingWorkflowId);
    }
  }, [pendingWorkflowId]);

  return null;
};

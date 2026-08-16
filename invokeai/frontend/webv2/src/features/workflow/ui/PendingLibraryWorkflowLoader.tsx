import { getLibraryWorkflow, touchLibraryWorkflowOpenedAt } from '@features/workflow/data/api';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import { parseWorkflowJson, serializeWorkflowJson } from '@features/workflow/utility';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useTranslation } from 'react-i18next';

import { markLibraryGraphSynced } from './library/librarySyncBridge';
import { startWorkflowUiPendingLoadRuntime } from './pendingLibraryWorkflowLoadRuntime';

/**
 * Consumes `pendingLibraryWorkflowId` requests from shell surfaces (the
 * command palette) that cannot reach the graph context themselves: fetches
 * the library record, parses it, and replaces the project graph — the same
 * load path as WorkflowLibraryDialog, minus the dialog.
 */
export const PendingLibraryWorkflowLoader = () => {
  const { t } = useTranslation();
  const { replace } = useProjectGraphCommands();
  const notify = useWorkflowNotifications();
  useMountEffect(() => {
    const owner = captureAccountScope();

    return startWorkflowUiPendingLoadRuntime(async (workflowId) => {
      try {
        assertAccountScopeCurrent(owner);
        const raw = await getLibraryWorkflow(workflowId, owner.signal);

        assertAccountScopeCurrent(owner);
        const { document, warnings } = parseWorkflowJson(raw);
        const name = typeof raw.name === 'string' && raw.name.length > 0 ? raw.name : 'workflow';

        replace(document, t('commandPalette.workflowLoad.loaded', { name }));
        // Same reasoning as the library dialog's load path: the graph just
        // loaded is already in sync with the library record it came from, so
        // mark it synced before the autosaver's graph-changed effect sees it.
        markLibraryGraphSynced(serializeWorkflowJson(document));

        for (const warning of warnings) {
          notify.info(t('commandPalette.workflowLoad.warning'), warning);
        }

        void touchLibraryWorkflowOpenedAt(workflowId, owner.signal).catch(() => {
          // Recency bookkeeping only; loading already succeeded.
        });
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        notify.error(
          t('commandPalette.workflowLoad.failed'),
          getApiErrorMessage(error, t('commandPalette.workflowLoad.couldNotLoad'))
        );
      }
    });
  });

  return null;
};

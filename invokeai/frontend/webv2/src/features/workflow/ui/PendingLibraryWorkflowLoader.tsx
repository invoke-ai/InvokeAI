import { getLibraryWorkflow, touchLibraryWorkflowOpenedAt } from '@features/workflow/data/api';
import { updateLoadedWorkflowNodes } from '@features/workflow/data/templates';
import { requestWorkflowFitView } from '@features/workflow/ui/editor/flowInstanceStore';
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
 * Apply external library or embedded-workflow requests through the same parse-and-replace path as the library
 * dialog.
 */
export const PendingWorkflowLoader = () => {
  const { t } = useTranslation();
  const { replace } = useProjectGraphCommands();
  const notify = useWorkflowNotifications();
  useMountEffect(() => {
    const owner = captureAccountScope();

    return startWorkflowUiPendingLoadRuntime(async (source) => {
      try {
        assertAccountScopeCurrent(owner);

        let raw: unknown = null;
        let label = '';

        if (source.kind === 'library') {
          const record = await getLibraryWorkflow(source.workflowId, owner.signal);
          const name = typeof record.name === 'string' && record.name.length > 0 ? record.name : 'workflow';

          raw = record;
          label = t('commandPalette.workflowLoad.loaded', { name });
        } else {
          raw = source.raw;
          label = source.label;
        }

        assertAccountScopeCurrent(owner);
        const { document: parsed, warnings: parseWarnings } = parseWorkflowJson(raw);
        const { document: updated, warnings: updateWarnings } = updateLoadedWorkflowNodes(parsed, t);
        const warnings = [...parseWarnings, ...updateWarnings];

        // An embedded document may still carry the id of the library record it was
        // saved from; loading it must not start autosaving over that record.
        const document = source.kind === 'library' ? updated : { ...updated, libraryWorkflowId: undefined };

        replace(document, label);
        requestWorkflowFitView(document.nodes);

        if (source.kind === 'library') {
          // Mark freshly loaded library graphs synced before autosave observes them as edits.
          markLibraryGraphSynced(serializeWorkflowJson(document));
          void touchLibraryWorkflowOpenedAt(source.workflowId, owner.signal).catch(() => {
            // Recency bookkeeping only; loading already succeeded.
          });
        }

        for (const warning of warnings) {
          notify.info(t('commandPalette.workflowLoad.warning'), warning);
        }
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

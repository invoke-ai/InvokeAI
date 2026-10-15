import { getLibraryWorkflowRecord, touchLibraryWorkflowOpenedAt } from '@features/workflow/data/api';
import { updateLoadedWorkflowNodes } from '@features/workflow/data/templates';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowNotifications, useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import { requestLibraryCopyChoice } from '@features/workflow/ui/workflowUiStore';
import { parseWorkflowJson } from '@features/workflow/utility';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useTranslation } from 'react-i18next';

import { planLibraryWorkflowOpen } from './library/useOpenLibraryWorkflow';
import { startWorkflowUiPendingLoadRuntime } from './pendingLibraryWorkflowLoadRuntime';

/**
 * Apply external library or embedded-workflow requests through the same collection commands as the library
 * dialog: a template becomes the project's first copy, or the copy-choice host asks what to do once it has one;
 * an embedded document becomes a new workflow.
 */
export const PendingWorkflowLoader = () => {
  const { t } = useTranslation();
  const { addWorkflow } = useProjectGraphCommands();
  const { project } = useWorkflowUi();
  const notify = useWorkflowNotifications();
  useMountEffect(() => {
    const owner = captureAccountScope();

    return startWorkflowUiPendingLoadRuntime(async (source) => {
      try {
        assertAccountScopeCurrent(owner);

        // The request is fenced to the project it arrived in.
        const projectId = project.getSnapshot().id;

        if (source.kind === 'library') {
          const plan = planLibraryWorkflowOpen(project.getSnapshot().workflows, source.workflowId);

          if (plan.kind === 'choose') {
            requestLibraryCopyChoice(projectId, {
              // Without the template's name, the copy's own stands in for it.
              name: source.name ?? plan.copies[0]!.document.name,
              workflow_id: source.workflowId,
            });
            return;
          }
        }

        let raw: unknown = null;
        let label = '';
        let libraryRevision: number | null = null;

        if (source.kind === 'library') {
          const record = await getLibraryWorkflowRecord(source.workflowId, owner.signal);
          const name = record.name.length > 0 ? record.name : 'workflow';

          raw = { ...record.workflow, id: record.workflow_id };
          label = t('commandPalette.workflowLoad.loaded', { name });
          libraryRevision = record.revision;
        } else {
          raw = source.raw;
          label = source.label;
        }

        assertAccountScopeCurrent(owner);

        if (project.getSnapshot().id !== projectId) {
          return;
        }

        const { document: parsed, warnings: parseWarnings } = parseWorkflowJson(raw);
        const { document, warnings: updateWarnings } = updateLoadedWorkflowNodes(parsed, t);
        const warnings = [...parseWarnings, ...updateWarnings];

        addWorkflow(document, {
          label,
          reusePlaceholder: true,
          // An embedded document may carry the id of the record it was saved from; that id is provenance, never
          // a write target.
          ...(source.kind === 'library'
            ? { source: { libraryWorkflowId: source.workflowId, revision: libraryRevision } }
            : {}),
        });

        if (source.kind === 'library') {
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

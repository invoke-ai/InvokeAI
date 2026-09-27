import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';

import { updateLoadedWorkflowNodes } from '@features/workflow/data/templates';
import { getLibraryWorkflowRecordCached, touchLibraryWorkflowOpenedAt } from '@features/workflow/queries';
import { requestWorkflowFitView } from '@features/workflow/ui/editor/flowInstanceStore';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowNotifications, useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import { parseWorkflowJson } from '@features/workflow/utility';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { findProjectCopiesOf } from './projectWorkflowEntries';

/**
 * Opening a template always lands in the project's own collection: as a new copy, or by resuming the copy the
 * project already made from it. Overlapping opens are dropped; a queued second load would add a second copy.
 */

export type WorkflowLoadPhase = 'applying' | 'fetching' | 'idle';

export type OpenLibraryWorkflowMode =
  /** Resume the one existing copy, or add the first; several copies need a choice the caller makes. */
  | 'resume-or-add'
  /** Always add a fresh, independent copy beside any existing ones. */
  | 'add-copy';

/** What opening a template would do given the project's current copies. */
export type LibraryWorkflowOpenPlan =
  | { kind: 'add' }
  | { kind: 'resume'; workflowId: string }
  | { kind: 'choose'; copies: readonly ProjectWorkflowEntry[] };

export const planLibraryWorkflowOpen = (
  workflows: readonly ProjectWorkflowEntry[],
  libraryWorkflowId: string
): LibraryWorkflowOpenPlan => {
  const copies = findProjectCopiesOf(workflows, libraryWorkflowId);

  if (copies.length === 0) {
    return { kind: 'add' };
  }

  if (copies.length === 1) {
    return { kind: 'resume', workflowId: copies[0]!.document.id };
  }

  return { copies, kind: 'choose' };
};

export interface OpenLibraryWorkflow {
  /** Fetches the template and adds a copy, or resumes the existing one; resolves once the project has it. */
  open: (item: Pick<WorkflowLibraryListItem, 'name' | 'workflow_id'>, mode: OpenLibraryWorkflowMode) => Promise<void>;
  /** Activates a copy the project already owns. */
  resume: (workflowId: string) => void;
  /** Drives the caller's busy overlay; `applying` is the expensive half. */
  loadPhase: WorkflowLoadPhase;
}

export const useOpenLibraryWorkflow = (onOpened: () => void): OpenLibraryWorkflow => {
  const { t } = useTranslation();
  const { addWorkflow, selectWorkflow } = useProjectGraphCommands();
  const { project } = useWorkflowUi();
  const notify = useWorkflowNotifications();
  const [loadPhase, setLoadPhase] = useState<WorkflowLoadPhase>('idle');
  const isInFlightRef = useRef(false);

  const resume = useCallback(
    (workflowId: string) => {
      selectWorkflow(workflowId);
      onOpened();
    },
    [onOpened, selectWorkflow]
  );

  const open = useCallback(
    async (item: Pick<WorkflowLibraryListItem, 'name' | 'workflow_id'>, mode: OpenLibraryWorkflowMode) => {
      const owner = captureAccountScope();

      if (isInFlightRef.current) {
        return;
      }

      if (mode === 'resume-or-add') {
        const plan = planLibraryWorkflowOpen(project.getSnapshot().workflows, item.workflow_id);

        if (plan.kind === 'resume') {
          resume(plan.workflowId);
          return;
        }

        if (plan.kind === 'choose') {
          // Several copies exist; the surface offering the open owns the choice.
          return;
        }
      }

      isInFlightRef.current = true;

      // The project the open started from is the project it lands in, whatever becomes active meanwhile.
      const projectId = project.getSnapshot().id;

      try {
        setLoadPhase('fetching');

        const record = await getLibraryWorkflowRecordCached(item.workflow_id, owner.signal);

        assertAccountScopeCurrent(owner);
        const { document: parsed, warnings: parseWarnings } = parseWorkflowJson({
          ...record.workflow,
          id: record.workflow_id,
        });
        const { document, warnings: updateWarnings } = updateLoadedWorkflowNodes(parsed, t);
        const warnings = [...parseWarnings, ...updateWarnings];

        assertAccountScopeCurrent(owner);
        setLoadPhase('applying');
        // Adding a large document is synchronous and expensive; give React a frame to paint the busy overlay.
        await new Promise<void>((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
        });

        assertAccountScopeCurrent(owner);

        if (project.getSnapshot().id !== projectId) {
          notify.info(t('workflowLibrary.openProjectChanged', { name: item.name }));
          return;
        }

        addWorkflow(document, {
          label: t('workflowLibrary.loadedLabel', { name: item.name }),
          // The first copy may take over the blank a fresh project starts with; another copy never does.
          reusePlaceholder: mode === 'resume-or-add',
          source: { libraryWorkflowId: record.workflow_id, revision: record.revision },
        });
        requestWorkflowFitView(document.nodes);

        for (const warning of warnings) {
          notify.info(t('workflowLibrary.loadWarning'), warning);
        }

        void touchLibraryWorkflowOpenedAt(item.workflow_id, owner.signal).catch(() => {
          // Recency bookkeeping only; opening already succeeded.
        });
        onOpened();
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        notify.error(
          t('workflowLibrary.loadFailed'),
          getApiErrorMessage(error, t('workflowLibrary.loadFailedBody', { name: item.name }))
        );
      } finally {
        if (isAccountScopeCurrent(owner)) {
          isInFlightRef.current = false;
          setLoadPhase('idle');
        }
      }
    },
    [addWorkflow, notify, onOpened, project, resume, t]
  );

  return { loadPhase, open, resume };
};

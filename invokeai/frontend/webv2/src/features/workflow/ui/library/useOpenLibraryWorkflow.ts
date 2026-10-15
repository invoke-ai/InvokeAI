import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { LibraryOpenItem } from '@features/workflow/ui/workflowUiStore';

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
 * Opening a template always lands in the project's own collection. The first copy is added directly; once the
 * project holds one, the caller asks whether to open a copy, add another, or replace one with the library version.
 * Overlapping loads are dropped; a queued second load would add a second copy.
 */

export type WorkflowLoadPhase = 'applying' | 'fetching' | 'idle';

export type OpenLibraryWorkflowMode =
  /** The project's first copy; it may take over the untouched blank a fresh project starts with. */
  | 'first-copy'
  /** Another, independent copy beside the existing ones. */
  | 'add-copy';

/** What opening a template involves given the project's current copies. */
export type LibraryWorkflowOpenPlan = { kind: 'add' } | { kind: 'choose'; copies: readonly ProjectWorkflowEntry[] };

export const planLibraryWorkflowOpen = (
  workflows: readonly ProjectWorkflowEntry[],
  libraryWorkflowId: string
): LibraryWorkflowOpenPlan => {
  const copies = findProjectCopiesOf(workflows, libraryWorkflowId);

  return copies.length === 0 ? { kind: 'add' } : { copies, kind: 'choose' };
};

export interface OpenLibraryWorkflow {
  /** Fetches the template and adds a copy; resolves once the project has it. */
  open: (item: LibraryOpenItem, mode: OpenLibraryWorkflowMode) => Promise<void>;
  /** Fetches the template and puts it in place of an existing copy, as one undo step. */
  replace: (item: LibraryOpenItem, workflowId: string) => Promise<void>;
  /** Activates a copy the project already owns. */
  resume: (workflowId: string) => void;
  /** Drives the caller's busy overlay; `applying` is the expensive half. */
  loadPhase: WorkflowLoadPhase;
}

type LibraryLoadTarget = { kind: 'add'; mode: OpenLibraryWorkflowMode } | { kind: 'replace'; workflowId: string };

export const useOpenLibraryWorkflow = (onOpened: () => void): OpenLibraryWorkflow => {
  const { t } = useTranslation();
  const { addWorkflow, replaceWorkflow, selectWorkflow } = useProjectGraphCommands();
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

  const load = useCallback(
    async (item: LibraryOpenItem, target: LibraryLoadTarget) => {
      const owner = captureAccountScope();

      if (isInFlightRef.current) {
        return;
      }

      isInFlightRef.current = true;

      // The project the open started from is the project it lands in, whatever becomes active meanwhile.
      const projectId = project.getSnapshot().id;

      try {
        setLoadPhase('fetching');

        const record = await getLibraryWorkflowRecordCached(item.workflow_id, {
          expectedRevision: item.revision,
          signal: owner.signal,
        });

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

        const source = { libraryWorkflowId: record.workflow_id, revision: record.revision };

        if (target.kind === 'replace') {
          replaceWorkflow({ projectId, workflowId: target.workflowId }, document, {
            label: t('workflowLibrary.replacedLabel', { name: item.name }),
            source,
          });
          // The copy keeps its id, so its editor stays mounted and fits the new graph on request.
          requestWorkflowFitView({ projectId, workflowId: target.workflowId }, document.nodes);
        } else {
          addWorkflow(document, {
            label: t('workflowLibrary.loadedLabel', { name: item.name }),
            reusePlaceholder: target.mode === 'first-copy',
            source,
          });
        }

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
    [addWorkflow, notify, onOpened, project, replaceWorkflow, t]
  );
  const open = useCallback(
    (item: LibraryOpenItem, mode: OpenLibraryWorkflowMode) => load(item, { kind: 'add', mode }),
    [load]
  );
  const replace = useCallback(
    (item: LibraryOpenItem, workflowId: string) => load(item, { kind: 'replace', workflowId }),
    [load]
  );

  return { loadPhase, open, replace, resume };
};

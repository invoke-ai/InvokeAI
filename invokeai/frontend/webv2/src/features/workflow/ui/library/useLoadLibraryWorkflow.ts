import type { WorkflowLibraryListItem } from '@features/workflow/queries';

import { getLibraryWorkflowCached, touchLibraryWorkflowOpenedAt } from '@features/workflow/queries';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import { parseWorkflowJson, serializeWorkflowJson } from '@features/workflow/utility';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { markLibraryGraphSynced } from './librarySyncBridge';

/**
 * The library's "open this workflow into the project graph" sequence, shared
 * by every surface that can start it (the library grid's double-click and its
 * Open action). Two loads can never overlap — a second call while one is in
 * flight is dropped rather than queued, because the winner would silently
 * replace the graph the first one just applied.
 */

export type WorkflowLoadPhase = 'applying' | 'fetching' | 'idle';

export interface LoadLibraryWorkflow {
  load: (item: WorkflowLibraryListItem) => Promise<void>;
  /** Drives the caller's busy overlay; `applying` is the expensive half. */
  loadPhase: WorkflowLoadPhase;
}

export const useLoadLibraryWorkflow = (onLoaded: () => void): LoadLibraryWorkflow => {
  const { t } = useTranslation();
  const { replace } = useProjectGraphCommands();
  const notify = useWorkflowNotifications();
  const [loadPhase, setLoadPhase] = useState<WorkflowLoadPhase>('idle');
  const isInFlightRef = useRef(false);

  const load = useCallback(
    async (item: WorkflowLibraryListItem): Promise<void> => {
      const owner = captureAccountScope();

      if (isInFlightRef.current) {
        return;
      }
      isInFlightRef.current = true;

      try {
        setLoadPhase('fetching');

        const raw = await getLibraryWorkflowCached(item.workflow_id, owner.signal);

        assertAccountScopeCurrent(owner);
        const { document, warnings } = parseWorkflowJson(raw);

        assertAccountScopeCurrent(owner);
        setLoadPhase('applying');
        // Synchronous graph replacement can be expensive for large workflows.
        // Give React a full frame to commit and paint the busy overlay first.
        await new Promise<void>((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
        });

        assertAccountScopeCurrent(owner);
        replace(document, t('workflowLibrary.loadedLabel', { name: item.name }));
        // The freshly-loaded graph is already in sync with the library record
        // it came from — mark it synced so the autosaver does not immediately
        // queue a redundant (echo) save the moment the graph reference changes.
        markLibraryGraphSynced(serializeWorkflowJson(document));

        for (const warning of warnings) {
          notify.info(t('workflowLibrary.loadWarning'), warning);
        }

        void touchLibraryWorkflowOpenedAt(item.workflow_id, owner.signal).catch(() => {
          // Recency bookkeeping only; loading already succeeded.
        });
        onLoaded();
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
    [notify, onLoaded, replace, t]
  );

  return { load, loadPhase };
};

import type { WorkflowLibraryListItem } from '@features/workflow/queries';

import { updateLoadedWorkflowNodes } from '@features/workflow/data/templates';
import { getLibraryWorkflowCached, touchLibraryWorkflowOpenedAt } from '@features/workflow/queries';
import { requestWorkflowFitView } from '@features/workflow/ui/editor/flowInstanceStore';
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

/** Drop overlapping workflow-open requests; a queued second load would silently replace the first applied graph. */

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
        const { document: parsed, warnings: parseWarnings } = parseWorkflowJson(raw);
        const { document, warnings: updateWarnings } = updateLoadedWorkflowNodes(parsed, t);
        const warnings = [...parseWarnings, ...updateWarnings];

        assertAccountScopeCurrent(owner);
        setLoadPhase('applying');
        // Synchronous graph replacement can be expensive for large workflows.
        // Give React a full frame to commit and paint the busy overlay first.
        await new Promise<void>((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
        });

        assertAccountScopeCurrent(owner);
        replace(document, t('workflowLibrary.loadedLabel', { name: item.name }));
        requestWorkflowFitView(document.nodes);
        // Mark loaded content synced before changed graph identity can schedule an echo save.
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

import {
  createLibraryWorkflow,
  invalidateWorkflowLibraryCache,
  updateLibraryWorkflow,
} from '@features/workflow/queries';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowNotifications, useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { serializeWorkflowJson } from '@features/workflow/utility';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useCallback } from 'react';

import { markLibraryGraphSynced } from './librarySyncBridge';
import { setWorkflowLibrarySyncStatus } from './workflowLibrarySyncStore';

/**
 * Saves the project graph to the backend workflow library — shared by the
 * library dialog's Save/Save as new buttons and the header's one-click save
 * for unbound graphs. A successful save also marks the autosaver's baseline
 * (`markLibraryGraphSynced`) so the freshly-saved content is not immediately
 * queued for another autosave pass.
 */
export const useSaveWorkflowToLibrary = (): {
  saveAsNew: () => Promise<string | null>;
  saveToLibrary: () => Promise<string | null>;
} => {
  const projectGraph = useWorkflowProjectSelector((project) => project.projectGraph);
  const { bindLibraryWorkflow } = useProjectGraphCommands();
  const notify = useWorkflowNotifications();

  const save = useCallback(
    async (asNew: boolean): Promise<string | null> => {
      const owner = captureAccountScope();

      try {
        const serialized = serializeWorkflowJson(projectGraph);
        let workflowId: string;

        if (!asNew && projectGraph.libraryWorkflowId) {
          workflowId = projectGraph.libraryWorkflowId;
          await updateLibraryWorkflow(workflowId, serialized, owner.signal);

          assertAccountScopeCurrent(owner);
          notify.success('Workflow saved', `Updated "${projectGraph.name || 'Untitled Workflow'}" in the library.`);
        } else {
          workflowId = await createLibraryWorkflow(serialized, owner.signal);

          assertAccountScopeCurrent(owner);
          bindLibraryWorkflow(workflowId);
          notify.success('Workflow saved', `Saved "${projectGraph.name || 'Untitled Workflow'}" to the library.`);
        }

        markLibraryGraphSynced(serialized);
        setWorkflowLibrarySyncStatus('saved');
        invalidateWorkflowLibraryCache();

        return workflowId;
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return null;
        }

        notify.error('Failed to save workflow', getApiErrorMessage(error, 'The workflow could not be saved.'));
        return null;
      }
    },
    [bindLibraryWorkflow, notify, projectGraph]
  );

  const saveToLibrary = useCallback(() => save(false), [save]);
  const saveAsNew = useCallback(() => save(true), [save]);

  return { saveAsNew, saveToLibrary };
};

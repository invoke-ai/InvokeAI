import type { ProjectGraphState } from '@features/workflow/core/types';

import {
  createLibraryWorkflow,
  invalidateWorkflowLibraryCache,
  updateLibraryWorkflow,
} from '@features/workflow/queries';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  useWorkflowNotifications,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from '@features/workflow/ui/WorkflowUiContext';
import { serializeWorkflowJson } from '@features/workflow/utility';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { markLibraryGraphSynced } from './librarySyncBridge';
import { setWorkflowLibrarySyncStatus } from './workflowLibrarySyncStore';

/**
 * Saves the project graph to the backend workflow library — the chrome
 * header's one-click save for unbound graphs, and the graph preview's "save
 * this document as a new entry". A successful save also marks the autosaver's
 * baseline (`markLibraryGraphSynced`) so the freshly-saved content is not
 * immediately queued for another autosave pass.
 */
export const useSaveWorkflowToLibrary = (): {
  saveDocumentAsNew: (document: ProjectGraphState) => Promise<string | null>;
  saveToLibrary: () => Promise<string | null>;
} => {
  const projectGraph = useWorkflowProjectSelector((project) => project.projectGraph);
  const { project: projectStore } = useWorkflowUi();
  const { bindLibraryWorkflow } = useProjectGraphCommands();
  const notify = useWorkflowNotifications();
  const { t } = useTranslation();

  const saveToLibrary = useCallback(async (): Promise<string | null> => {
    const owner = captureAccountScope();

    try {
      const serialized = serializeWorkflowJson(projectGraph);
      let workflowId: string;
      let syncedSerialized = serialized;

      const name = projectGraph.name || t('workflowLibrary.untitled');

      if (projectGraph.libraryWorkflowId) {
        workflowId = projectGraph.libraryWorkflowId;
        await updateLibraryWorkflow(workflowId, serialized, owner.signal);

        assertAccountScopeCurrent(owner);
        notify.success(t('workflowLibrary.saved'), t('workflowLibrary.savedUpdatedBody', { name }));
      } else {
        workflowId = await createLibraryWorkflow(serialized, owner.signal);

        assertAccountScopeCurrent(owner);
        bindLibraryWorkflow(workflowId);
        notify.success(t('workflowLibrary.saved'), t('workflowLibrary.savedCreatedBody', { name }));

        // bindLibraryWorkflow dispatches synchronously, so the store already
        // reflects the bound `libraryWorkflowId`. Re-serialize from that
        // post-bind graph (rather than reusing the pre-bind `serialized`,
        // which has no `id`) so the autosaver's synced baseline matches
        // exactly what its own read() will produce next — otherwise the id
        // key `serializeWorkflowJson` adds on bind reads as a dirty edit and
        // triggers a redundant echo save on the next debounce.
        syncedSerialized = serializeWorkflowJson(projectStore.getSnapshot().projectGraph);
      }

      markLibraryGraphSynced(syncedSerialized);
      setWorkflowLibrarySyncStatus('saved');
      invalidateWorkflowLibraryCache();

      return workflowId;
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return null;
      }

      notify.error(t('workflowLibrary.saveFailed'), getApiErrorMessage(error, t('common.unknownError')));
      return null;
    }
  }, [bindLibraryWorkflow, notify, projectGraph, projectStore, t]);

  // Saves an arbitrary document — not necessarily the active project graph —
  // as a new library entry. Used by "Open as → Save to workflow library" for
  // a preview payload that may never have become the active project. Unlike
  // `saveToLibrary`, this never binds the result to the project or marks the
  // autosaver's synced baseline: the active project graph is left alone.
  const saveDocumentAsNew = useCallback(
    async (document: ProjectGraphState): Promise<string | null> => {
      const owner = captureAccountScope();

      try {
        const serialized = serializeWorkflowJson(document);
        const workflowId = await createLibraryWorkflow(serialized, owner.signal);

        assertAccountScopeCurrent(owner);
        invalidateWorkflowLibraryCache();

        return workflowId;
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return null;
        }

        notify.error(t('workflowLibrary.saveFailed'), getApiErrorMessage(error, t('common.unknownError')));
        return null;
      }
    },
    [notify, t]
  );

  return { saveDocumentAsNew, saveToLibrary };
};

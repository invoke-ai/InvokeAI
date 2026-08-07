import type { Project } from '@workbench/projectContracts';

import { flushGenerateDrafts } from '@features/generation/react';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useNavigate } from '@tanstack/react-router';
import { useNotify } from '@workbench/useNotify';
import {
  useWorkbenchCommands,
  useWorkbenchPersistenceAdapter,
  useWorkbenchPersistenceService,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { useTranslation } from 'react-i18next';

import { deleteLibraryProject, refreshProjectLibrary } from './library';

/**
 * Open, close, and delete for projects, shared by the top bar and the Project
 * panel so the semantics stay in one place:
 *
 * - Open switches to the project when the session already has it, and otherwise
 *   hydrates it from the server first. Callers name a project; whether it is
 *   already loaded is not their problem.
 *
 * - Close flushes the document, drops the tab, and keeps the project in the
 *   library. Closing the last tab persists the empty session and lands on
 *   Home — an editor with no documents is the Home screen.
 * - Delete removes the project from the server (the only path that does, for
 *   open projects) and then closes its tab.
 */
export const useProjectActions = (): {
  closeProject: (project: Project) => void;
  deleteProject: (project: Project) => Promise<void>;
  openProject: (projectId: string, name: string) => Promise<void>;
} => {
  const queries = useWorkbenchQueries();
  const persistence = useWorkbenchPersistenceAdapter();
  const persistenceService = useWorkbenchPersistenceService();
  const commands = useWorkbenchCommands();
  const navigate = useNavigate();
  const notify = useNotify();
  const { t } = useTranslation();

  /** When the last tab goes, the session empties and Home takes over. */
  const leaveEditorIfLast = (projectId: string): boolean => {
    if (queries.getSnapshot().projects.some((project) => project.id !== projectId)) {
      return false;
    }

    void persistenceService.persistEmptySession(persistence.getState());
    void navigate({ to: '/' });

    return true;
  };

  const openProject = async (projectId: string, name: string): Promise<void> => {
    const owner = captureAccountScope();

    flushGenerateDrafts();

    if (queries.getSnapshot().projects.some((project) => project.id === projectId)) {
      commands.projects.switchTo(projectId);

      return;
    }

    try {
      const project = await persistenceService.hydrateProjectFromServer(projectId);

      assertAccountScopeCurrent(owner);

      if (!project) {
        notify.error(t('projects.couldNotOpen'), t('projects.couldNotOpenDescription', { name }));
        void refreshProjectLibrary();

        return;
      }

      commands.projects.open(project);
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      notify.error(
        t('projects.couldNotOpen'),
        getApiErrorMessage(error, t('projects.couldNotOpenDescription', { name }))
      );
    }
  };

  const closeProject = (project: Project): void => {
    flushGenerateDrafts();

    const projectToFlush = queries.getProject(project.id) ?? project;

    // `.finally()` forwards the rejection it was chained onto, so `void` alone left an unhandled
    // one behind — reachable by closing a tab while the account is going away, which is when the
    // flush rejects. The tab is closing either way; the flush was best-effort.
    void persistenceService
      .flushProjectToServer(projectToFlush)
      .finally(() => {
        persistenceService.releaseProjectSync(project.id);
      })
      .catch(() => undefined);

    if (leaveEditorIfLast(project.id)) {
      return;
    }

    commands.projects.close(project.id);
  };

  const deleteProject = async (project: Project): Promise<void> => {
    flushGenerateDrafts();

    try {
      // Deletion goes through the library for every surface. For a project the workbench holds it
      // routes through this editor's own sync handle, which is what stops an in-flight autosave
      // recreating the project — and with it a board — between the DELETE and the tab closing.
      await deleteLibraryProject(project.id);
    } catch (error) {
      persistenceService.unmarkProjectDeleted(project.id);
      notify.error(t('projects.deleteFailed'), error instanceof Error ? error.message : undefined);

      return;
    }

    if (leaveEditorIfLast(project.id)) {
      return;
    }

    commands.projects.close(project.id);
  };

  return { closeProject, deleteProject, openProject };
};

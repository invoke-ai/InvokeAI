import type { Project } from '@workbench/types';

import { useNavigate } from '@tanstack/react-router';
import { useNotify } from '@workbench/useNotify';
import { useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';

import { deleteLibraryProject } from './library';
import {
  flushProjectToServer,
  markProjectDeleted,
  persistEmptySession,
  releaseProjectSync,
  unmarkProjectDeleted,
} from './syncedPersistence';

/**
 * Close and delete for projects that are open in the editor, shared by the
 * tab bar and the Project panel so the semantics stay in one place:
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
} => {
  const state = useWorkbenchSelector((snapshot) => snapshot.state);
  const dispatch = useWorkbenchDispatch();
  const navigate = useNavigate();
  const notify = useNotify();

  /** When the last tab goes, the session empties and Home takes over. */
  const leaveEditorIfLast = (projectId: string): boolean => {
    if (state.projects.some((project) => project.id !== projectId)) {
      return false;
    }

    void persistEmptySession(state);
    void navigate({ to: '/' });

    return true;
  };

  const closeProject = (project: Project): void => {
    void flushProjectToServer(project).finally(() => {
      releaseProjectSync(project.id);
    });

    if (leaveEditorIfLast(project.id)) {
      return;
    }

    dispatch({ projectId: project.id, type: 'closeProject' });
  };

  const deleteProject = async (project: Project): Promise<void> => {
    // Marked before the request so an in-flight autosave cannot recreate the
    // project server-side between the DELETE and the tab closing.
    markProjectDeleted(project.id);

    try {
      await deleteLibraryProject(project.id);
    } catch (error) {
      unmarkProjectDeleted(project.id);
      notify.error('Delete failed', error instanceof Error ? error.message : undefined);

      return;
    }

    if (leaveEditorIfLast(project.id)) {
      return;
    }

    dispatch({ projectId: project.id, type: 'closeProject' });
  };

  return { closeProject, deleteProject };
};

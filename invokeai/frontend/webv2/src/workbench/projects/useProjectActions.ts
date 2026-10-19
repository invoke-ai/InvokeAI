import type { GalleryBoardDeletionResult } from '@features/gallery';
import type { Project } from '@workbench/projectContracts';

import { getGalleryProjectBoardsFromCaches, invalidateGallery } from '@features/gallery/queries';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from '@tanstack/react-router';
import { hasActiveQueueRuns } from '@workbench/queue-integration/activeQueueRuns';
import { useNotify } from '@workbench/useNotify';
import {
  useWorkbenchLiveCanvasEngines,
  useWorkbenchCommands,
  useWorkbenchPersistenceAdapter,
  useWorkbenchPersistenceService,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { useTranslation } from 'react-i18next';

import type { DeleteProjectBoards } from './api';

import { deleteLibraryProject, refreshProjectLibrary } from './library';
import { serializeProjectDocumentV3Json } from './projectDocument';
import { describeRefusedProject } from './projectLoadRefusal';

const CLOSE_FLUSH_ATTEMPTS = 3;

/** A project's boards go without their media, which returns to Uncategorized. */
const NO_MEDIA_DELETED: Omit<GalleryBoardDeletionResult, 'boardId'> = {
  deletedBoardImageNames: [],
  deletedBoardVideoNames: [],
  deletedImageNames: [],
  deletedVideoNames: [],
  failedImageNames: [],
  failedVideoNames: [],
};

/**
 * Open reuses or hydrates a project. Close flushes while preserving the server record; closing the last tab
 * returns Home. Delete removes the server project.
 */
export const useProjectActions = (): {
  closeProject: (project: Project) => void;
  deleteProject: (project: Project, boards?: DeleteProjectBoards) => Promise<void>;
  openProject: (projectId: string, name: string) => Promise<void>;
} => {
  const queries = useWorkbenchQueries();
  const persistence = useWorkbenchPersistenceAdapter();
  const persistenceService = useWorkbenchPersistenceService();
  const commands = useWorkbenchCommands();
  const canvasEngines = useWorkbenchLiveCanvasEngines();
  const navigate = useNavigate();
  const notify = useNotify();
  const { t } = useTranslation();
  const queryClient = useQueryClient();

  /** False when the project changed since `unchangedFrom` (an edit was still pending), so it must be pushed again. */
  const finishClose = async (projectId: string, unchangedFrom?: Project): Promise<boolean> => {
    const closeResult = commands.projects.close(projectId, unchangedFrom);
    if (closeResult.ok || closeResult.reason === 'project-not-found') {
      persistenceService.releaseProjectSync(projectId);
      return true;
    }
    if (closeResult.reason === 'modified') {
      return false;
    }
    if (closeResult.reason === 'active-queue-runs') {
      throw new Error(t('projects.activeRunsMustFinish'));
    }
    if (closeResult.reason !== 'last-project') {
      throw new Error(t('projects.file.notSynced'));
    }

    let hasLeftEditor = false;
    try {
      await persistenceService.persistEmptySession(persistence.getState());
      const retry = commands.projects.close(projectId, unchangedFrom);
      if (retry.ok || retry.reason === 'project-not-found') {
        persistenceService.releaseProjectSync(projectId);
        return true;
      }
      if (retry.reason === 'modified') {
        return false;
      }
      if (retry.reason === 'active-queue-runs') {
        throw new Error(t('projects.activeRunsMustFinish'));
      }
      if (retry.reason !== 'last-project') {
        throw new Error(t('projects.file.notSynced'));
      }

      persistenceService.releaseProjectSync(projectId);
      await navigate({ to: '/' });
      hasLeftEditor = true;
      return true;
    } finally {
      if (!hasLeftEditor) {
        // The editor stays open after all, so its session must name its projects again.
        void persistenceService.reopenSession(persistence.getState()).catch(() => undefined);
      }
    }
  };

  const openProject = async (projectId: string, name: string): Promise<void> => {
    const owner = captureAccountScope();

    if (queries.getSnapshot().projects.some((project) => project.id === projectId)) {
      commands.projects.switchTo(projectId);

      return;
    }

    try {
      const result = await persistenceService.hydrateProjectFromServer(projectId, name);

      assertAccountScopeCurrent(owner);

      if (result.status === 'refused') {
        const notice = describeRefusedProject(result.refused, t);

        notify.error(notice.title, notice.message);

        return;
      }

      if (result.status !== 'loaded') {
        notify.error(t('projects.couldNotOpen'), t('projects.couldNotOpenDescription', { name }));
        void refreshProjectLibrary();

        return;
      }

      commands.projects.open(result.project);
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
    const owner = captureAccountScope();

    if (hasActiveQueueRuns(queries.getProject(project.id) ?? project)) {
      notify.error(t('projects.closeBlocked'), t('projects.activeRunsMustFinish'));
      return;
    }

    /** Crosses the canvas paint barrier; false (with a notice) when unsaved pixels cannot be persisted. */
    const persistCanvasPixels = async (): Promise<boolean> => {
      try {
        await canvasEngines.flushPendingPixels(project.id);
        return true;
      } catch (error) {
        assertAccountScopeCurrent(owner);
        notify.error(
          t('projects.closeBlocked'),
          t('projects.canvasPixelsNotSaved', { reason: getApiErrorMessage(error, t('projects.file.notSynced')) })
        );
        return false;
      }
    };

    void (async () => {
      for (let attempt = 0; attempt < CLOSE_FLUSH_ATTEMPTS; attempt += 1) {
        const current = queries.getProject(project.id);
        if (!current) {
          return;
        }
        if (!(await persistCanvasPixels())) {
          return;
        }
        assertAccountScopeCurrent(owner);
        const outcome = await persistenceService.flushProjectToServer(queries.getProject(project.id) ?? current);
        assertAccountScopeCurrent(owner);
        if (outcome.kind === 'schema-refused') {
          notify.error(t('projects.closeBlocked'), t('projects.file.updateClient'));
          return;
        }

        if (outcome.kind === 'unsynced' || outcome.kind === 'conflicted') {
          notify.error(t('projects.closeBlocked'), t('projects.file.notSynced'));
          return;
        }
        // Pixels painted while the document was pushed change it again, so the comparison below retries.
        if (!(await persistCanvasPixels())) {
          return;
        }
        const pushed = queries.getProject(project.id) ?? current;
        if (
          outcome.kind === 'acknowledged' &&
          serializeProjectDocumentV3Json(pushed).documentJson !== outcome.documentJson
        ) {
          continue;
        }
        // Closing commits drafts still pending in editors; one that changes the project sends it round again.
        if (await finishClose(project.id, pushed)) {
          return;
        }
      }
      notify.error(t('projects.closeBlocked'), t('projects.file.notSynced'));
    })().catch((error) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }
      notify.error(t('projects.closeBlocked'), getApiErrorMessage(error, t('projects.file.notSynced')));
    });
  };

  const deleteProject = async (project: Project, boards?: DeleteProjectBoards): Promise<void> => {
    if (hasActiveQueueRuns(project)) {
      notify.error(t('projects.deleteFailed'), t('projects.activeRunsMustFinish'));
      return;
    }

    const owner = captureAccountScope();
    // The boards that go with it, as the board lists know them: its inbox, and the rest when they go too. Read before
    // the delete, whose refresh drops them from the lists.
    const deletedBoardIds = getGalleryProjectBoardsFromCaches(queryClient, project.id)
      .filter((board) => board.isInbox || boards === 'delete')
      .map((board) => board.id);
    try {
      // Open projects delete through the sync engine so in-flight saves finish first.
      await deleteLibraryProject(project.id, boards);
    } catch (error) {
      notify.error(t('projects.deleteFailed'), error instanceof Error ? error.message : undefined);

      return;
    }
    // Other open projects may still select, auto-add to or queue results for those boards. Only this account's.
    for (const boardId of isAccountScopeCurrent(owner) ? deletedBoardIds : []) {
      commands.gallery.reconcileDeletedBoardOutcome({ ...NO_MEDIA_DELETED, boardId });
    }
    // Its boards were released or deleted with it; every board list on screen is stale either way.
    void invalidateGallery(queryClient);

    try {
      await finishClose(project.id);
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('projects.deleteFailed'), getApiErrorMessage(error, t('projects.file.notSynced')));
      }
    }
  };

  return { closeProject, deleteProject, openProject };
};

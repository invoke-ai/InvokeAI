import type { Project } from '@workbench/projectContracts';

import { flushGenerateDrafts } from '@features/generation/react';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useNavigate } from '@tanstack/react-router';
import { hasActiveQueueRuns } from '@workbench/queue-integration/activeQueueRuns';
import { useNotify } from '@workbench/useNotify';
import {
  useWorkbenchCommands,
  useWorkbenchPersistenceAdapter,
  useWorkbenchPersistenceService,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { useTranslation } from 'react-i18next';

import { deleteLibraryProject, refreshProjectLibrary } from './library';
import { serializeProjectDocumentV3Json } from './projectDocument';
import { describeRefusedProject } from './projectLoadRefusal';

const CLOSE_FLUSH_ATTEMPTS = 3;

/**
 * Open reuses or hydrates a project. Close flushes while preserving the server record; closing the last tab
 * returns Home. Delete removes the server project.
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

  const finishClose = async (projectId: string): Promise<void> => {
    const closeResult = commands.projects.close(projectId);
    if (closeResult.ok || closeResult.reason === 'project-not-found') {
      persistenceService.releaseProjectSync(projectId);
      return;
    }
    if (closeResult.reason === 'active-queue-runs') {
      throw new Error(t('projects.activeRunsMustFinish'));
    }
    if (closeResult.reason !== 'last-project') {
      throw new Error(t('projects.file.notSynced'));
    }

    await persistenceService.persistEmptySession(persistence.getState());
    const retry = commands.projects.close(projectId);
    if (retry.ok || retry.reason === 'project-not-found') {
      persistenceService.releaseProjectSync(projectId);
      return;
    }
    if (retry.reason === 'active-queue-runs') {
      throw new Error(t('projects.activeRunsMustFinish'));
    }
    if (retry.reason !== 'last-project') {
      throw new Error(t('projects.file.notSynced'));
    }

    persistenceService.releaseProjectSync(projectId);
    await navigate({ to: '/' });
  };

  const openProject = async (projectId: string, name: string): Promise<void> => {
    const owner = captureAccountScope();

    flushGenerateDrafts();

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
    flushGenerateDrafts();

    if (hasActiveQueueRuns(queries.getProject(project.id) ?? project)) {
      notify.error(t('projects.closeBlocked'), t('projects.activeRunsMustFinish'));
      return;
    }

    void (async () => {
      for (let attempt = 0; attempt < CLOSE_FLUSH_ATTEMPTS; attempt += 1) {
        const current = queries.getProject(project.id);
        if (!current) {
          return;
        }
        const outcome = await persistenceService.flushProjectToServer(current);
        assertAccountScopeCurrent(owner);
        if (outcome.kind === 'schema-refused') {
          notify.error(t('projects.closeBlocked'), t('projects.file.updateClient'));
          return;
        }

        if (outcome.kind === 'unsynced' || outcome.kind === 'conflicted') {
          notify.error(t('projects.closeBlocked'), t('projects.file.notSynced'));
          return;
        }
        if (
          outcome.kind === 'acknowledged' &&
          serializeProjectDocumentV3Json(queries.getProject(project.id) ?? current).documentJson !==
            outcome.documentJson
        ) {
          continue;
        }
        await finishClose(project.id);
        return;
      }
      notify.error(t('projects.closeBlocked'), t('projects.file.notSynced'));
    })().catch((error) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }
      notify.error(t('projects.closeBlocked'), getApiErrorMessage(error, t('projects.file.notSynced')));
    });
  };

  const deleteProject = async (project: Project): Promise<void> => {
    flushGenerateDrafts();

    if (hasActiveQueueRuns(project)) {
      notify.error(t('projects.deleteFailed'), t('projects.activeRunsMustFinish'));
      return;
    }

    const owner = captureAccountScope();
    try {
      // Open projects delete through the sync engine so in-flight saves finish first.
      await deleteLibraryProject(project.id);
    } catch (error) {
      notify.error(t('projects.deleteFailed'), error instanceof Error ? error.message : undefined);

      return;
    }

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

import type { DeleteProjectBoards } from '@workbench/projects/api';

import { toaster } from '@platform/ui';
import { useQueryClient } from '@tanstack/react-query';
import { deleteLibraryProject, renameLibraryProject, type ProjectSummary } from '@workbench/projects/library';
import { useDuplicateProject, useExportLibraryProject } from '@workbench/projects/useProjectFileActions';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { dropProjectPin } from './projectPins';

/** Share server-backed library actions between grid and list without mounting the editor. */
export interface ProjectCardActions {
  rename: (name: string) => Promise<void>;
  /** Reports its own progress and result, so there is nothing to await here. */
  duplicate: () => void;
  /** Reports its own progress and result, so there is nothing to await here. */
  export: () => void;
  delete: (boards?: DeleteProjectBoards) => Promise<void>;
}

export const useProjectCardActions = (summary: ProjectSummary): ProjectCardActions => {
  const { t } = useTranslation();
  const startExport = useExportLibraryProject();
  const startDuplicate = useDuplicateProject();
  const queryClient = useQueryClient();

  const rename = useCallback(
    async (name: string) => {
      try {
        await renameLibraryProject(summary.id, name);
      } catch (error) {
        toaster.create({
          description: error instanceof Error ? error.message : undefined,
          title: t('projects.renameFailed'),
          type: 'error',
        });
        // The rename dialog stays open on a rejection, so the user keeps their input.
        throw error;
      }
    },
    [summary.id, t]
  );

  // Use the import/export reporter because duplication shares their restore engine and progress/error semantics.
  const duplicate = useCallback(() => {
    startDuplicate(summary.id);
  }, [startDuplicate, summary.id]);

  const exportProject = useCallback(() => {
    startExport(summary.id, summary.name);
  }, [startExport, summary.id, summary.name]);

  const deleteProject = useCallback(
    async (boards?: DeleteProjectBoards) => {
      try {
        await deleteLibraryProject(summary.id, boards);
        // Pins are persisted per account, so a deleted project would otherwise
        // leave a dead id in preferences forever.
        dropProjectPin(summary.id);
        // Its boards were released or deleted with it; any board list on screen is stale either way. Loaded here
        // rather than at the top (the gallery data layer is not part of the Launchpad's initial graph), and never
        // awaited: the project is gone, so a chunk that fails to load must not be reported as a failed delete.
        void import('@features/gallery/queries')
          .then((module) => module.invalidateGallery(queryClient))
          .catch(() => undefined);
      } catch (error) {
        toaster.create({
          description: error instanceof Error ? error.message : undefined,
          title: t('projects.deleteFailed'),
          type: 'error',
        });
      }
    },
    [queryClient, summary.id, t]
  );

  // Stabilize the actions object so virtualized row renders do not invalidate derived menu callbacks.
  return useMemo(
    () => ({ delete: deleteProject, duplicate, export: exportProject, rename }),
    [deleteProject, duplicate, exportProject, rename]
  );
};

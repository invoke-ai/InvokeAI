import { toaster } from '@platform/ui';
import {
  deleteLibraryProject,
  duplicateLibraryProject,
  renameLibraryProject,
  type ProjectSummary,
} from '@workbench/projects/library';
import { useExportLibraryProject } from '@workbench/projects/useProjectFileActions';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { dropProjectPin } from './projectPins';

/**
 * The library actions a single project offers, shared by the grid card and the
 * list row so the two cannot drift. Every one runs against the server without
 * mounting the editor.
 */
export interface ProjectCardActions {
  rename: (name: string) => Promise<void>;
  duplicate: () => Promise<void>;
  /** Reports its own progress and result, so there is nothing to await here. */
  export: () => void;
  delete: () => Promise<void>;
}

export const useProjectCardActions = (summary: ProjectSummary): ProjectCardActions => {
  const { t } = useTranslation();
  const startExport = useExportLibraryProject();

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

  const duplicate = useCallback(async () => {
    try {
      const copy = await duplicateLibraryProject(summary.id);

      toaster.create({
        description: t('projects.projectDuplicatedDescription', { name: copy.name }),
        title: t('projects.projectDuplicated'),
        type: 'success',
      });
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: t('projects.duplicateFailed'),
        type: 'error',
      });
    }
  }, [summary.id, t]);

  const exportProject = useCallback(() => {
    startExport(summary.id, summary.name);
  }, [startExport, summary.id, summary.name]);

  const deleteProject = useCallback(async () => {
    try {
      await deleteLibraryProject(summary.id);
      // Pins are persisted per account, so a deleted project would otherwise
      // leave a dead id in preferences forever.
      dropProjectPin(summary.id);
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: t('projects.deleteFailed'),
        type: 'error',
      });
    }
  }, [summary.id, t]);

  return { delete: deleteProject, duplicate, export: exportProject, rename };
};

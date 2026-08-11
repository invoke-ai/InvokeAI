import { toaster } from '@platform/ui';
import { deleteLibraryProject, renameLibraryProject, type ProjectSummary } from '@workbench/projects/library';
import { useDuplicateProject, useExportLibraryProject } from '@workbench/projects/useProjectFileActions';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { dropProjectPin } from './projectPins';

/**
 * The library actions a single project offers, shared by the grid card and the
 * list row so the two cannot drift. Every one runs against the server without
 * mounting the editor.
 */
export interface ProjectCardActions {
  rename: (name: string) => Promise<void>;
  /** Reports its own progress and result, so there is nothing to await here. */
  duplicate: () => void;
  /** Reports its own progress and result, so there is nothing to await here. */
  export: () => void;
  delete: () => Promise<void>;
}

export const useProjectCardActions = (summary: ProjectSummary): ProjectCardActions => {
  const { t } = useTranslation();
  const startExport = useExportLibraryProject();
  const startDuplicate = useDuplicateProject();

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

  // Progress, partial success and error translation all come from the shared reporter — the same
  // one import and export use. Duplication runs the same restore engine over the same board, so a
  // hand-rolled toast here only meant it reported that work differently from its twins.
  const duplicate = useCallback(() => {
    startDuplicate(summary.id);
  }, [startDuplicate, summary.id]);

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

  // Memoized because `ProjectActionsMenu` derives callbacks from this object, and the browser
  // renders one of these per row in a virtualized list: a fresh literal each render invalidates
  // every one of them on every render.
  return useMemo(
    () => ({ delete: deleteProject, duplicate, export: exportProject, rename }),
    [deleteProject, duplicate, exportProject, rename]
  );
};

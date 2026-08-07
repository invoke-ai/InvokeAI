import type { Project } from '@workbench/projectContracts';

import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { ProjectRecordDTO } from './api';
import type { DuplicatedProject } from './library';
import type { ProjectFileDirection } from './projectFileErrors';

import { duplicateLibraryProject } from './library';
import { exportLibraryProject, exportOpenProject, importProjectFile, pickProjectFile } from './projectFile';
import { startProjectFileReport } from './projectFileToasts';

/**
 * Picking, importing and exporting a project file, with the reporting attached.
 *
 * Five surfaces offer these — Home, the Projects page, the in-editor Open
 * dialog, the project switcher, and every project card — and each had written
 * out the same twenty lines of scope capture, picker, try/catch and toast. They
 * had already drifted: the import sites checked the account scope before
 * showing an error and the export sites did not, so cancelling an export by
 * signing out produced a toast written for a developer.
 *
 * Keeping the sequence in one place is what makes progress and partial-success
 * reporting land on all five at once rather than on whichever was edited last.
 * What differs between call sites is only what happens with the imported record,
 * which is the callback.
 */

/** Runs the sequence, keeping the toast and the account scope in step. */
const runReported = async <T>(
  t: (key: string, options?: Record<string, unknown>) => string,
  titles: { direction?: ProjectFileDirection; failed: string; running: string },
  run: (report: ReturnType<typeof startProjectFileReport>, owner: ReturnType<typeof captureAccountScope>) => Promise<T>
): Promise<void> => {
  const owner = captureAccountScope();
  const report = startProjectFileReport(t, titles.running, titles.direction);

  try {
    await run(report, owner);
  } catch (error) {
    // An operation cancelled by the account going away failed on purpose, and
    // its error is written for a developer. There is no verdict to show,
    // because there is no longer anyone the verdict is about.
    if (!isAccountScopeCurrent(owner)) {
      report.dismiss();

      return;
    }

    report.fail(titles.failed, error);
  }
};

/**
 * Import from the file picker. Resolves without doing anything if the picker is
 * dismissed — no toast is opened for a decision not to import.
 */
export const useImportProjectFile = (onImported: (record: ProjectRecordDTO) => Promise<void> | void): (() => void) => {
  const { t } = useTranslation();

  const importFile = useCallback(async () => {
    const owner = captureAccountScope();
    const file = await pickProjectFile(owner);

    if (!file || !isAccountScopeCurrent(owner)) {
      return;
    }

    await runReported(t, { failed: t('projects.importFailed'), running: t('projects.importing') }, async (report) => {
      const { record, ...issues } = await importProjectFile(file, { onProgress: report.report, owner });

      assertAccountScopeCurrent(owner);
      report.succeed(t('projects.imported', { name: record.name }), issues);
      await onImported(record);
      assertAccountScopeCurrent(owner);
    });
  }, [onImported, t]);

  return useCallback(() => void importFile(), [importFile]);
};

/** Export a project that is only a library row — its document comes from the server. */
export const useExportLibraryProject = (): ((projectId: string, name: string) => void) => {
  const { t } = useTranslation();

  return useCallback(
    (projectId: string, name: string) => {
      void runReported(
        t,
        { direction: 'write', failed: t('projects.exportFailed'), running: t('projects.exporting', { name }) },
        async (report, owner) => {
          const issues = await exportLibraryProject(projectId, { onProgress: report.report, owner });

          report.succeed(t('projects.exported', { name }), issues);
        }
      );
    },
    [t]
  );
};

/**
 * Duplicate a project, reported like the transfers it shares its engine with.
 *
 * Duplication is a project file operation in everything but the file: same restore, same partial
 * success, same durations. It reported none of that — no progress for the whole server-side copy,
 * and `InvkFormatError`'s developer message straight into a toast on failure.
 */
export const useDuplicateProject = (
  onDuplicated?: (duplicated: DuplicatedProject) => Promise<void> | void
): ((projectId: string) => void) => {
  const { t } = useTranslation();

  return useCallback(
    (projectId: string) => {
      void runReported(
        t,
        { direction: 'write', failed: t('projects.duplicateFailed'), running: t('projects.duplicating') },
        async (report, owner) => {
          const duplicated = await duplicateLibraryProject(projectId, {
            // Everything a duplication moves is restored onto the copy's board, so the phase is
            // fixed. Naming it here keeps the reporting vocabulary in the reporting layer.
            onProgress: ({ completed, total }) => report.report({ completed, phase: 'restoring', total }),
            owner,
          });

          assertAccountScopeCurrent(owner);
          report.succeed(t('projects.projectDuplicated'), duplicated);
          await onDuplicated?.(duplicated);
          assertAccountScopeCurrent(owner);
        }
      );
    },
    [onDuplicated, t]
  );
};

/** Export a project that is open in the editor, from its live document. */
export const useExportOpenProject = (): ((project: Project) => void) => {
  const { t } = useTranslation();

  return useCallback(
    (project: Project) => {
      void runReported(
        t,
        {
          direction: 'write',
          failed: t('projects.exportFailed'),
          running: t('projects.exporting', { name: project.name }),
        },
        async (report, owner) => {
          const issues = await exportOpenProject(project, { onProgress: report.report, owner });

          report.succeed(t('projects.exported', { name: project.name }), issues);
        }
      );
    },
    [t]
  );
};

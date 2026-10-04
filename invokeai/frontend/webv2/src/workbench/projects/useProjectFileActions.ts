import type { Project } from '@workbench/projectContracts';

import { fontKeys } from '@features/fonts/contracts';
import { createLogger } from '@platform/logging/logger';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { ProjectRecordDTO } from './api';
import type { DuplicatedProject } from './library';
import type { ProjectFileDirection } from './projectFileErrors';

import { useProjectFileOptions } from './components/ProjectFileOptionsProvider';
import { FontImportQuotaError } from './invk/format';
import { duplicateLibraryProject } from './library';
import { exportLibraryProject, exportOpenProject, importProjectFile, pickProjectFile } from './projectFile';
import { startProjectFileReport } from './projectFileToasts';

/** Share the scoped transfer and reporting lifecycle across callers. */

const projectFileLogger = createLogger({ area: 'project-file', namespace: 'persistence' });

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
    projectFileLogger.info({
      context: { direction: titles.direction, operation: titles.running },
      message: 'Project file operation completed',
      name: 'persistence.project-file-completed',
    });
  } catch (error) {
    // Account-cancellation errors stay silent because the operation no longer belongs to the active session.
    if (!isAccountScopeCurrent(owner)) {
      report.dismiss();

      return;
    }

    projectFileLogger.error({
      context: { direction: titles.direction },
      error,
      message: titles.failed,
      name: 'persistence.project-file-failed',
    });
    report.fail(titles.failed, error);
  }
};

/** Picker dismissal resolves without work or a toast. */
export const useImportProjectFile = (onImported: (record: ProjectRecordDTO) => Promise<void> | void): (() => void) => {
  const { t } = useTranslation();
  const { requestReferencesOnlyImport } = useProjectFileOptions();
  const queryClient = useQueryClient();

  const importFile = useCallback(async () => {
    const owner = captureAccountScope();
    const file = await pickProjectFile(owner);

    if (!file || !isAccountScopeCurrent(owner)) {
      return;
    }

    await runReported(t, { failed: t('projects.importFailed'), running: t('projects.importing') }, async (report) => {
      let outcome;
      try {
        outcome = await importProjectFile(file, { onProgress: report.report, owner });
      } catch (error) {
        if (!(error instanceof FontImportQuotaError) || !isAccountScopeCurrent(owner)) {
          throw error;
        }
        if (!(await requestReferencesOnlyImport(owner))) {
          report.dismiss();
          return;
        }
        outcome = await importProjectFile(file, { onProgress: report.report, owner, skipEmbeddedFonts: true });
      }
      const { record, ...issues } = outcome;

      assertAccountScopeCurrent(owner);
      report.succeed(t('projects.imported', { name: record.name }), issues);
      await onImported(record);
      assertAccountScopeCurrent(owner);
    });
    if (isAccountScopeCurrent(owner)) {
      void queryClient.invalidateQueries({ queryKey: fontKeys.all });
    }
  }, [onImported, queryClient, requestReferencesOnlyImport, t]);

  return useCallback(() => void importFile(), [importFile]);
};

/** Export a project that is only a library row — its document comes from the server. */
export const useExportLibraryProject = (): ((projectId: string, name: string) => void) => {
  const { t } = useTranslation();
  const { requestExportOptions } = useProjectFileOptions();

  return useCallback(
    (projectId: string, name: string) => {
      const owner = captureAccountScope();
      void requestExportOptions(name, owner).then(async (options) => {
        if (!options || !isAccountScopeCurrent(owner)) {
          return;
        }
        await runReported(
          t,
          { direction: 'write', failed: t('projects.exportFailed'), running: t('projects.exporting', { name }) },
          async (report) => {
            const issues = await exportLibraryProject(projectId, { ...options, onProgress: report.report, owner });

            report.succeed(t('projects.exported', { name }), issues);
          }
        );
      });
    },
    [requestExportOptions, t]
  );
};

/** Duplication shares transfer progress and partial-success reporting. */
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
            // Duplication uses the restore phase; this owner defines its reporting vocabulary.
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
  const { requestExportOptions } = useProjectFileOptions();

  return useCallback(
    (project: Project) => {
      const owner = captureAccountScope();
      void requestExportOptions(project.name, owner).then(async (options) => {
        if (!options || !isAccountScopeCurrent(owner)) {
          return;
        }
        await runReported(
          t,
          {
            direction: 'write',
            failed: t('projects.exportFailed'),
            running: t('projects.exporting', { name: project.name }),
          },
          async (report) => {
            const issues = await exportOpenProject(project, { ...options, onProgress: report.report, owner });

            report.succeed(t('projects.exported', { name: project.name }), issues);
          }
        );
      });
    },
    [requestExportOptions, t]
  );
};

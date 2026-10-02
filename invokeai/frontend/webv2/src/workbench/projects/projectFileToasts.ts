import { toaster } from '@platform/ui';

import type { InvkMediaIssueReason, ProjectTransferIssues } from './invk/transfer';
import type { ProjectFileProgress } from './projectFile';

import { describeProjectFileError, type ProjectFileDirection } from './projectFileErrors';

/** Update one toast with asset counts, not byte progress; partial loss completes as a warning. */

/** i18next's `t`, narrowed to what this module needs. */
type Translate = (key: string, options?: Record<string, unknown>) => string;

export interface ProjectFileReporter {
  /** Account loss dismisses the toast without a verdict. */
  dismiss: () => void;
  /** Update the live toast from a transfer's progress. */
  report: (progress: ProjectFileProgress) => void;
  /** Count missing board media separately from missing document references. */
  succeed: (title: string, issues: ProjectTransferIssues) => void;
  /** Finish as a failure, translating an `InvkFormatError` reason where there is one. */
  fail: (title: string, error: unknown) => void;
}

/** Exclude star-failed from missing-media counts because the bytes arrived. */
const MISSING_REASONS: ReadonlySet<InvkMediaIssueReason> = new Set(['fetch-failed', 'missing-entry', 'upload-failed']);

const countIssues = (issues: ProjectTransferIssues) => ({
  boardItems: issues.boardItemIssues.filter((issue) => MISSING_REASONS.has(issue.reason)).length,
  documentReferences: issues.documentReferenceIssues.filter((issue) => MISSING_REASONS.has(issue.reason)).length,
  unstarred: issues.boardItemIssues.filter((issue) => issue.reason === 'star-failed').length,
});

const describeProgress = (t: Translate, progress: ProjectFileProgress): string => {
  if (progress.phase === 'packing') {
    return t('projects.file.packing');
  }
  if (progress.phase === 'restoring-fonts') {
    return t('projects.fonts.installing', { completed: progress.completed, total: progress.total });
  }

  const key = progress.phase === 'bundling' ? 'projects.file.bundlingProgress' : 'projects.file.restoringProgress';

  return t(key, { completed: progress.completed, total: progress.total });
};

/** Progress arrives once per asset; five redraws a second is as much as anyone reads. */
const PROGRESS_REDRAW_INTERVAL_MS = 200;

/**
 * Every reporter path must finalize the toast. Direction determines failure wording; account cancellation
 * dismisses without a verdict.
 */
export const startProjectFileReport = (
  t: Translate,
  title: string,
  direction: ProjectFileDirection = 'read'
): ProjectFileReporter => {
  const id = toaster.create({
    // Keep the toast alive until the operation completes.
    duration: Number.POSITIVE_INFINITY,
    title,
    type: 'loading',
  });

  let isSettled = false;
  let redrawTimer: ReturnType<typeof setTimeout> | null = null;
  let pendingDescription: string | null = null;

  const clearRedraw = (): void => {
    if (redrawTimer !== null) {
      clearTimeout(redrawTimer);
      redrawTimer = null;
    }
    pendingDescription = null;
  };

  const settle = (options: { description?: string; title: string; type: 'error' | 'success' | 'warning' }): void => {
    isSettled = true;
    clearRedraw();
    toaster.update(id, { ...options, duration: undefined });
  };

  return {
    dismiss: () => {
      isSettled = true;
      clearRedraw();
      toaster.dismiss(id);
    },
    fail: (failureTitle, error) => {
      // Success is final even if follow-up navigation fails.
      if (isSettled) {
        return;
      }

      const description = describeProjectFileError(error, t, direction);

      settle({ ...(description === undefined ? {} : { description }), title: failureTitle, type: 'error' });
    },
    report: (progress) => {
      if (isSettled) {
        return;
      }

      pendingDescription = describeProgress(t, progress);

      if (redrawTimer !== null) {
        return;
      }

      // Publish the leading update immediately and retain the latest trailing count.
      toaster.update(id, { description: pendingDescription });
      pendingDescription = null;
      redrawTimer = setTimeout(() => {
        redrawTimer = null;

        if (pendingDescription !== null && !isSettled) {
          toaster.update(id, { description: pendingDescription });
          pendingDescription = null;
        }
      }, PROGRESS_REDRAW_INTERVAL_MS);
    },
    succeed: (successTitle, issues) => {
      const { boardItems, documentReferences, unstarred } = countIssues(issues);

      if (boardItems === 0 && documentReferences === 0 && unstarred === 0) {
        settle({ title: successTitle, type: 'success' });

        return;
      }

      // Keep toasts bounded to counts; detailed losses remain on the outcome.
      const parts = [
        ...(boardItems === 0 ? [] : [t('projects.file.missingBoardItems', { count: boardItems })]),
        ...(documentReferences === 0 ? [] : [t('projects.file.missingReferences', { count: documentReferences })]),
        ...(unstarred === 0 ? [] : [t('projects.file.unstarredItems', { count: unstarred })]),
      ];

      settle({ description: parts.join(' '), title: successTitle, type: 'warning' });
    },
  };
};

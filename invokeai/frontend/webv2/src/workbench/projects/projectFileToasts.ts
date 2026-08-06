import { toaster } from '@platform/ui';

import type { ProjectTransferIssues } from './invk/transfer';
import type { ProjectFileProgress } from './projectFile';

import { describeProjectFileError } from './projectFileErrors';

/**
 * How a project file reports itself while it runs, and what it says when it is
 * done.
 *
 * Every entry point — Home, the Projects page, the in-editor Open dialog, the
 * project switcher, each project card — used to start one of these and then say
 * nothing until it either finished or threw. For an operation whose duration is
 * set by how much someone has drawn, that is a long silence: a few hundred
 * full-resolution layers is a few hundred round trips and hundreds of megabytes.
 * A button that appears to do nothing for two minutes gets pressed again.
 *
 * The reporter is one live toast rather than a new one per phase, so a slow
 * export occupies one line and updates in place instead of stacking. It is
 * deliberately unglamorous: a count, then a result. There is no progress bar,
 * because the honest unit here is assets and the expensive one is bytes, and a
 * bar drawn from the first would sit at 99% through the whole of the second.
 *
 * ### Half-success is a result, not a success
 *
 * Export skips assets the server will not serve; import leaves references
 * dangling when neither the archive nor this server has them. Both are correct
 * — a project that exports every layer but one is far more use than one that
 * refuses — but both are things the person holding the file needs to know
 * before they hand it to someone else. So a lossy run finishes as a warning
 * naming the count, and only a clean one finishes as a plain success.
 */

/** i18next's `t`, narrowed to what this module needs. */
type Translate = (key: string, options?: Record<string, unknown>) => string;

export interface ProjectFileReporter {
  /**
   * Take the toast down without a verdict, for the one case that has none: the
   * account went away, so the operation belongs to a session that is over.
   */
  dismiss: () => void;
  /** Update the live toast from a transfer's progress. */
  report: (progress: ProjectFileProgress) => void;
  /**
   * Finish cleanly, or as a warning naming what was left behind.
   *
   * Board items and document references are counted apart because they cost different things: a
   * missing board item is a result still findable elsewhere, a missing document reference is a
   * hole in the canvas. One combined number meant anything from "you will not notice" to "the
   * project is broken".
   */
  succeed: (title: string, issues: ProjectTransferIssues) => void;
  /** Finish as a failure, translating an `InvkFormatError` reason where there is one. */
  fail: (title: string, error: unknown) => void;
}

const describeProgress = (t: Translate, progress: ProjectFileProgress): string => {
  if (progress.phase === 'packing') {
    return t('projects.file.packing');
  }

  const key = progress.phase === 'bundling' ? 'projects.file.bundlingProgress' : 'projects.file.restoringProgress';

  return t(key, { completed: progress.completed, total: progress.total });
};

/**
 * Open the live toast for a transfer that is starting. The returned reporter
 * owns that toast for the rest of the operation; every path through a caller's
 * `try`/`catch` has to end in `succeed` or `fail`, or the toast stays up.
 */
export const startProjectFileReport = (t: Translate, title: string): ProjectFileReporter => {
  const id = toaster.create({
    // No duration: a transfer ends when it ends, and a toast that expired
    // mid-export would leave the operation invisible again.
    duration: Number.POSITIVE_INFINITY,
    title,
    type: 'loading',
  });

  const settle = (options: { description?: string; title: string; type: 'error' | 'success' | 'warning' }): void => {
    toaster.update(id, { ...options, duration: undefined });
  };

  return {
    dismiss: () => {
      toaster.dismiss(id);
    },
    fail: (failureTitle, error) => {
      const description = describeProjectFileError(error, t);

      settle({ ...(description === undefined ? {} : { description }), title: failureTitle, type: 'error' });
    },
    report: (progress) => {
      toaster.update(id, { description: describeProgress(t, progress) });
    },
    succeed: (successTitle, issues) => {
      const boardCount = issues.boardItemIssues.length;
      const referenceCount = issues.documentReferenceIssues.length;

      if (boardCount === 0 && referenceCount === 0) {
        settle({ title: successTitle, type: 'success' });

        return;
      }

      // Counts, never names: a project can lose hundreds of assets at once, and a toast listing
      // them would be unreadable. The typed detail stays on the outcome for anything that needs it.
      const parts = [
        ...(boardCount === 0 ? [] : [t('projects.file.missingBoardItems', { count: boardCount })]),
        ...(referenceCount === 0 ? [] : [t('projects.file.missingReferences', { count: referenceCount })]),
      ];

      settle({ description: parts.join(' '), title: successTitle, type: 'warning' });
    },
  };
};

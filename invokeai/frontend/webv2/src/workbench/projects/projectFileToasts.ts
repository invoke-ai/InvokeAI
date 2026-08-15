import { toaster } from '@platform/ui';

import type { InvkMediaIssueReason, ProjectTransferIssues } from './invk/transfer';
import type { ProjectFileProgress } from './projectFile';

import { describeProjectFileError, type ProjectFileDirection } from './projectFileErrors';

/**
 * One live toast per transfer, updated in place. No progress bar: the countable unit is assets and
 * the expensive one is bytes, so a bar drawn from the first would sit at 99% through the second.
 *
 * A lossy run settles as a warning naming the count, not a success — the person holding the file
 * needs to know before they hand it to someone else.
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
   * Board items and document references are counted apart: a missing board item is a result still
   * findable elsewhere, a missing document reference is a hole in the canvas.
   */
  succeed: (title: string, issues: ProjectTransferIssues) => void;
  /** Finish as a failure, translating an `InvkFormatError` reason where there is one. */
  fail: (title: string, error: unknown) => void;
}

/**
 * The reasons that mean the media is *not here*. `star-failed` is excluded: the item arrived, it
 * just did not get its star back, and counting it would report lost media when nothing was lost.
 */
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

  const key = progress.phase === 'bundling' ? 'projects.file.bundlingProgress' : 'projects.file.restoringProgress';

  return t(key, { completed: progress.completed, total: progress.total });
};

/** Progress arrives once per asset; five redraws a second is as much as anyone reads. */
const PROGRESS_REDRAW_INTERVAL_MS = 200;

/**
 * Open the live toast. The reporter owns it for the rest of the operation — every path through a
 * caller's `try`/`catch` must end in `succeed` or `fail`, or the toast stays up. `direction` words
 * the failure: the same `too-large` reason arises from opening a file and from exporting one.
 */
export const startProjectFileReport = (
  t: Translate,
  title: string,
  direction: ProjectFileDirection = 'read'
): ProjectFileReporter => {
  const id = toaster.create({
    // No duration: a transfer ends when it ends, and a toast that expired
    // mid-export would leave the operation invisible again.
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
      // A verdict is not taken back. `succeed` runs before the caller's own follow-up work — the
      // navigation after an import, say — and a failure there is not a failure of the transfer:
      // telling someone their import failed after it demonstrably worked is the worse lie.
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

      // Leading edge, then trailing: the first count appears at once, and whatever the latest is
      // when the interval expires replaces it.
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

      // Counts, never names: a project can lose hundreds of assets at once, and a toast listing
      // them would be unreadable. The typed detail stays on the outcome for anything that needs it.
      const parts = [
        ...(boardItems === 0 ? [] : [t('projects.file.missingBoardItems', { count: boardItems })]),
        ...(documentReferences === 0 ? [] : [t('projects.file.missingReferences', { count: documentReferences })]),
        ...(unstarred === 0 ? [] : [t('projects.file.unstarredItems', { count: unstarred })]),
      ];

      settle({ description: parts.join(' '), title: successTitle, type: 'warning' });
    },
  };
};

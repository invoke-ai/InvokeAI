import { toaster } from '@platform/ui';

import type { InvkMediaIssueReason, ProjectTransferIssues } from './invk/transfer';
import type { ProjectFileProgress } from './projectFile';

import { describeProjectFileError, type ProjectFileDirection } from './projectFileErrors';

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

/**
 * The reasons that mean the media is *not here*, as opposed to here without its flag.
 *
 * `star-failed` is the odd one out: the item arrived, it just did not get its star back. Folding it
 * into "could not be included" tells someone their project lost media when nothing was lost, which
 * is the one thing these counts exist to answer accurately.
 *
 * Phrasing a reason is this module's job, so the set lives here — `transfer.ts` owns the taxonomy
 * and is imported type-only, which also keeps it out of the two routes' initial chunks.
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

/**
 * How often the live toast may redraw while a transfer runs.
 *
 * Progress arrives once per asset, and a project can have hundreds — each one otherwise a toaster
 * state update, during the phase that is already saturating the network. A count that moves five
 * times a second is as much as anyone reads, and the final value is always drawn, so the number a
 * person is left looking at is never a stale one.
 */
const PROGRESS_REDRAW_INTERVAL_MS = 200;

/**
 * Open the live toast for a transfer that is starting. The returned reporter
 * owns that toast for the rest of the operation; every path through a caller's
 * `try`/`catch` has to end in `succeed` or `fail`, or the toast stays up.
 *
 * `direction` decides how a failure is worded: the same `too-large` reason is raised by opening a
 * file someone handed you and by exporting a project of your own, and those need different
 * sentences.
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

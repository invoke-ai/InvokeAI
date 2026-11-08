import type { ProjectLibrarySnapshot } from '@workbench/projects/library';
import type { WorkbenchSearch } from '@workbench/projects/session';
import type { StartupDestination } from '@workbench/settings/contracts';

import {
  beginStartupLaunch,
  hasStayedOnLaunchpad,
  leaveStartupLaunch,
  openStartupLaunch,
  settleStartupLaunch,
} from '@workbench/launchpad/startupLaunch';
import { isProjectSummaryCompatible } from '@workbench/projects/library';

/** How the page was loaded: its URL hash and its Navigation Timing type. */
export interface InitialPageLoad {
  hash: string;
  navigationType?: NavigationTimingType;
}

export interface StartupDependencies {
  /** Projects the saved session would reopen; null when that cannot be told. */
  peekOpenProjectIds: () => Promise<string[] | null>;
  /** Lists the account's projects into the library and returns its snapshot (summaries newest edit first). */
  loadProjectLibrary: () => Promise<ProjectLibrarySnapshot>;
  /** Starts downloading the editor so it overlaps the lookups. */
  preloadEditor: () => void;
  /** Tells the viewer the launch stayed home because the previous one never rendered the editor. */
  reportSkippedLaunch: () => void;
}

/** Redirect options for the editor route: a project, or (without one) the saved session's tabs. */
export interface StartupRedirect {
  replace: true;
  search: Pick<WorkbenchSearch, 'project'>;
  to: '/app';
}

export interface StartupRedirects {
  /** Called on every authenticated route load; returns where the launch goes instead of Home, or null. */
  take: (
    arrival: { pathname: string; preload: boolean },
    destination: StartupDestination
  ) => Promise<StartupRedirect | null>;
  /**
   * True once, for the editor entry a session-restoring launch redirect started: the launch's peek already found open
   * projects, so the editor route's empty-session guard need not fetch the session again.
   */
  takeOpenSessionProof: () => boolean;
}

/** The most recently edited project this build can open; null when there is none or the library is unavailable. */
const findRecentProjectId = (library: ProjectLibrarySnapshot): string | null =>
  library.status === 'ready' ? (library.summaries.find(isProjectSummaryCompatible)?.id ?? null) : null;

/**
 * Decides, once per page load, whether the app's launch skips the Launchpad. Only opening the bare URL (a bookmark,
 * the installed app's start URL) is a launch: deep links keep their destination, including `#/` from reloading Home.
 * The router leaves a bare URL bare on Home, so reloading or returning to it there is not a launch either. The first
 * authenticated arrival (not a preload) consumes the launch, so it survives sign-in (which lands on Home) while later
 * visits to Home (app menu, the editor's empty-session guard, the not-found fallback) never redirect again. A session
 * with open projects reopens them; otherwise the launch opens the most recently edited compatible project, if any.
 * The splash can call the launch off while it decides or opens, and a launch whose editor never rendered makes the
 * next one stay on the Launchpad, with a notice.
 */
export const createStartupRedirects = (
  { hash, navigationType }: InitialPageLoad,
  dependencies: StartupDependencies
): StartupRedirects => {
  let isLaunchPending =
    (hash === '' || hash === '#') && navigationType !== 'reload' && navigationType !== 'back_forward';
  let hasOpenSessionProof = false;

  return {
    take: async ({ pathname, preload }, destination) => {
      if (preload) {
        return null;
      }

      // The proof only reaches the redirect's own arrival; any other route load drops it.
      if (pathname !== '/app') {
        hasOpenSessionProof = false;
      }

      if (!isLaunchPending) {
        if (pathname !== '/app') {
          leaveStartupLaunch();
        }
        return null;
      }

      isLaunchPending = false;

      if (pathname !== '/' || destination === 'launchpad') {
        return null;
      }

      // A launch whose editor never rendered (a project that breaks it, say) must not trap every launch after it.
      if (!beginStartupLaunch()) {
        dependencies.reportSkippedLaunch();
        return null;
      }

      dependencies.preloadEditor();
      const openProjectIds = await dependencies.peekOpenProjectIds();

      if (hasStayedOnLaunchpad()) {
        return null;
      }

      if (openProjectIds !== null && openProjectIds.length > 0) {
        hasOpenSessionProof = true;
        openStartupLaunch(null);
        return { replace: true, search: {}, to: '/app' };
      }

      // An empty or unknown session (no blob yet) opens the most recent project, as the editor itself would without a
      // blob; an unavailable library leaves the launch on the Launchpad.
      const projectId = findRecentProjectId(await dependencies.loadProjectLibrary());

      if (projectId === null || hasStayedOnLaunchpad()) {
        settleStartupLaunch();
        return null;
      }

      openStartupLaunch(projectId);
      return { replace: true, search: { project: projectId }, to: '/app' };
    },
    takeOpenSessionProof: () => {
      const hadProof = hasOpenSessionProof;
      hasOpenSessionProof = false;
      return hadProof;
    },
  };
};

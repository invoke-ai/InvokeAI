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

export interface InitialPageLoad {
  hash: string;
  navigationType?: NavigationTimingType;
}

export interface StartupDependencies {
  /** Projects the saved session would reopen; null when that cannot be told. */
  peekOpenProjectIds: () => Promise<string[] | null>;
  loadProjectLibrary: () => Promise<ProjectLibrarySnapshot>;
  preloadEditor: () => void;
  reportSkippedLaunch: () => void;
}

export interface StartupRedirect {
  replace: true;
  search: Pick<WorkbenchSearch, 'project'>;
  to: '/app';
}

export interface StartupRedirects {
  take: (
    arrival: { pathname: string; preload: boolean },
    destination: StartupDestination
  ) => Promise<StartupRedirect | null>;
  /** True once, for an editor entry whose launch peek already found open projects. */
  takeOpenSessionProof: () => boolean;
}

const findRecentProjectId = (library: ProjectLibrarySnapshot): string | null =>
  library.status === 'ready' ? (library.summaries.find(isProjectSummaryCompatible)?.id ?? null) : null;

/**
 * Only a bare URL is a launch; the first authenticated arrival consumes it, so it survives sign-in, never re-fires.
 * A launch whose editor never rendered makes the next one stay on the Launchpad.
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

      // An empty or unknown session opens the most recent project; an unavailable library stays on the Launchpad.
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

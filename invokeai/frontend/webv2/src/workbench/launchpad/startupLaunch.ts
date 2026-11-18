import { captureAccountScope } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * A launch that skips the Launchpad to reopen the last work (the "Most recent project" startup), while it is under
 * way. The splash offers a way to stay on the Launchpad meanwhile, and a launch whose editor never rendered (it failed
 * to load, or crashed first) leaves a mark in the account's browser storage so the next launch does not try again.
 *
 * `beginStartupLaunch` when the launch starts deciding, `openStartupLaunch` when it redirects into the editor,
 * `settleStartupLaunch` whenever the editor renders (or the launch stays home after all), `leaveStartupLaunch` when
 * another route loads first, `stayOnLaunchpad` when the user opts out. The mark names what the launch opened, so
 * deleting that project (`forgetStartupLaunchProject`) clears it too.
 */

export type StartupLaunchPhase = 'idle' | 'deciding' | 'opening';

interface StartupLaunchSnapshot {
  phase: StartupLaunchPhase;
  /** Set by "Stay on Launchpad"; a decision still in flight then gives up its redirect. */
  stayed: boolean;
}

const MARK_KEY_PREFIX = 'invokeai:v7:webv2:startup-launch-pending';
/** The mark's value when the launch restored the saved session's tabs rather than one project. */
const SESSION_MARK = '*';

const store = createExternalStore<StartupLaunchSnapshot>({ phase: 'idle', stayed: false });

const markKey = (): string => `${MARK_KEY_PREFIX}${captureAccountScope().storageSuffix}`;

const readMark = (): string | null => {
  try {
    return window.localStorage.getItem(markKey());
  } catch {
    return null;
  }
};

const writeMark = (value: string | null): void => {
  try {
    if (value === null) {
      window.localStorage.removeItem(markKey());
    } else {
      window.localStorage.setItem(markKey(), value);
    }
  } catch {
    // Without storage the safety net is off: every launch tries again, as before it existed.
  }
};

/**
 * Starts the launch decision for the current account. Returns false, and forgets the mark, when that account's
 * previous launch never rendered the editor: this launch then stays on the Launchpad.
 */
export const beginStartupLaunch = (): boolean => {
  if (readMark() !== null) {
    writeMark(null);
    return false;
  }
  store.setSnapshot({ phase: 'deciding', stayed: false });
  return true;
};

/** Whether the user chose to stay on the Launchpad while the launch was deciding. */
export const hasStayedOnLaunchpad = (): boolean => store.getSnapshot().stayed;

/** The launch redirects into the editor, for `projectId` or (null) the saved session's tabs: marked until it renders. */
export const openStartupLaunch = (projectId: string | null): void => {
  writeMark(projectId ?? SESSION_MARK);
  store.patchSnapshot({ phase: 'opening' });
};

/**
 * The editor rendered, or the launch stays home after all. Clears the account's mark whatever this page's phase: a
 * reload during "Opening project", or another tab's launch, leaves a mark this render proves stale.
 */
export const settleStartupLaunch = (): void => {
  writeMark(null);
  if (store.getSnapshot().phase !== 'idle') {
    store.setSnapshot({ phase: 'idle', stayed: false });
  }
};

/** Another route loaded before the editor rendered: the splash's way out is gone, the mark stays. */
export const leaveStartupLaunch = (): void => {
  if (store.getSnapshot().phase !== 'idle') {
    store.setSnapshot({ phase: 'idle', stayed: false });
  }
};

/** The user opted out from the splash. An open the user abandoned is not a failed one. */
export const stayOnLaunchpad = (): void => {
  writeMark(null);
  store.setSnapshot({ phase: 'idle', stayed: true });
};

/** A deleted project can no longer fail to open: a mark naming it would only skip the next launch for nothing. */
export const forgetStartupLaunchProject = (projectId: string): void => {
  if (readMark() === projectId) {
    writeMark(null);
  }
};

export const useStartupLaunchPhase = (): StartupLaunchPhase => store.useSelector((snapshot) => snapshot.phase);

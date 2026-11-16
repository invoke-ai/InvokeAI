import type { QueueActiveSession, QueueProgressSession } from '@features/queue/contracts';
import type { ReactNode } from 'react';

import {
  getAutoSwitchFollowedLive,
  getQueueActiveSessions,
  getQueueProgressSessions,
  isAutoSwitchPinHeld,
  isGalleryProgressItem,
  pickAutoSwitch,
} from '@features/queue/contracts';
import { useActiveProgressTargets, useFollowedProgressTargets, useFramedProgressTargets } from '@features/queue/react';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { createContext, use, useMemo, useState } from 'react';

interface LivePreviewFollow {
  sessions: QueueActiveSession[];
  gallerySessions: QueueProgressSession[];
  pinnedSessionId: string | null;
  /** The session Auto-switch shows (`getAutoSwitchFollowedLive`); else null, and the Preview keeps the selected result. */
  followedSessionId: string | null;
  /**
   * The newest running session still without a frame of its own (loading its model, encoding its prompt) while no
   * running session is followed: what the Preview says it waits for.
   */
  startingSession: QueueActiveSession | null;
  /** Sets Auto-switch to In progress and pins `sessionId`: a tile click, or an arrow step onto a tile. */
  follow(sessionId: string): void;
  pin(sessionId: string): void;
  showAll(): void;
}

const LivePreviewFollowContext = createContext<LivePreviewFollow | null>(null);

/** Owned by the mounted Workbench, never persisted with project documents. */
export const LivePreviewFollowProvider = ({ children }: { children: ReactNode }) => {
  const { projectId, items, mode } = useActiveProjectSelector((project) => ({
    projectId: project.id,
    items: project.queue.items,
    mode: project.settings.viewerAutoSwitch,
  }));
  // Live frames show only on In progress; Newest switches to finished results alone.
  const enabled = mode === 'progress';
  const running = useActiveProgressTargets();
  const followed = useFollowedProgressTargets();
  const framed = useFramedProgressTargets();
  const sessions = useMemo(
    () => getQueueActiveSessions(items.filter(isGalleryProgressItem), running, followed, framed),
    [items, running, followed, framed]
  );
  const gallerySessions = useMemo(
    () => getQueueProgressSessions(items.filter(isGalleryProgressItem), sessions),
    [items, sessions]
  );
  const [selection, setSelection] = useState<{ projectId: string; sessionId: string | null }>({
    projectId,
    sessionId: null,
  });
  const isStale =
    selection.projectId !== projectId ||
    !enabled ||
    (selection.sessionId !== null && !isAutoSwitchPinHeld(sessions, selection.sessionId));
  if (isStale && (selection.sessionId !== null || selection.projectId !== projectId)) {
    setSelection({ projectId, sessionId: null });
  }
  const pinnedSessionId = isStale ? null : selection.sessionId;
  // Sessions are in backend start order, so the newest running one is followed, not the one whose frame came last.
  const followedSession = getAutoSwitchFollowedLive(mode, sessions, pinnedSessionId);
  const followedSessionId = followedSession?.id ?? null;
  const startingSession =
    enabled && followedSession?.state !== 'running'
      ? (sessions.filter((session) => session.state === 'running' && !session.hasOwnFrame).at(-1) ?? null)
      : null;
  const { account } = useWorkbenchCommands();
  const value = useMemo<LivePreviewFollow>(
    () => ({
      sessions,
      gallerySessions,
      pinnedSessionId,
      followedSessionId,
      startingSession,
      follow: (sessionId) => {
        const { mode: nextMode, pinnedId } = pickAutoSwitch({ id: sessionId, kind: 'live' });
        account.updateProjectPreferences({ viewerAutoSwitch: nextMode });
        setSelection({ projectId, sessionId: pinnedId });
      },
      pin: (sessionId) => setSelection({ projectId, sessionId }),
      showAll: () => setSelection({ projectId, sessionId: null }),
    }),
    [account, sessions, gallerySessions, followedSessionId, pinnedSessionId, projectId, startingSession]
  );
  return <LivePreviewFollowContext value={value}>{children}</LivePreviewFollowContext>;
};

export const useLivePreviewFollow = (): LivePreviewFollow => {
  const context = use(LivePreviewFollowContext);
  if (!context) {
    throw new Error('Live preview requires its Workbench provider.');
  }
  return context;
};

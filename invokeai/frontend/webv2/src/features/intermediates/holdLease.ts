import type { AccountScope } from '@platform/state/accountLifecycle';

import { createUuid } from '@platform/browser/randomUuid';
import { apiFetch, getHttpAuthToken } from '@platform/transport/http';

export interface HeldMediaNames {
  images: readonly string[];
  videos: readonly string[];
}

/** Mirrors the server's per-kind `max_length` on a hold request; names past it are not held. */
export const MAX_HOLD_NAMES_PER_KIND = 50_000;

/** The server keeps a lease for 15 minutes; refreshing well inside that survives a missed beat. */
const HEARTBEAT_MS = 5 * 60_000;
/** A tab back from a stretch long enough that throttled timers may have missed the heartbeat. */
const RESEND_ON_VISIBLE_AFTER_MS = HEARTBEAT_MS;
const CHANGE_DEBOUNCE_MS = 250;

const sameNames = (left: ReadonlySet<string>, right: ReadonlySet<string>): boolean =>
  left.size === right.size && [...left].every((name) => right.has(name));

const holdPath = (leaseId: string): string => `/api/v1/intermediates/holds/${encodeURIComponent(leaseId)}`;

/**
 * Keeps the media an open editor still needs (unsaved content, undo state) out of cleanup for as long as the tab is
 * alive, under one lease the server replaces atomically. A send the server did not acknowledge is forgotten, so the
 * next change, heartbeat or return to the foreground sends again while the server keeps the previous lease. The
 * lease is released on dispose and on `pagehide`, with `keepalive` so the release survives the page; the server's
 * 15-minute expiry bounds anything a crash leaves behind.
 */
export const startIntermediatesHoldLease = ({
  owner,
  read,
  subscribe,
}: {
  owner: AccountScope;
  /** Current names to hold, in any order; returning the previous object again means nothing changed. */
  read: () => HeldMediaNames;
  /** Notifies when `read` may return something new. */
  subscribe: (onChange: () => void) => () => void;
}): (() => void) => {
  const leaseId = createUuid();
  // The body the server last accepted, or null once a send failed or the lease was released.
  let acknowledged: string | null = null;
  let held = false;
  let disposed = false;
  let inFlight = false;
  let pending = false;
  let pendingRefresh = false;
  let lastSentAt = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let planned: { held: HeldMediaNames; images: Set<string>; body: string; videos: Set<string> } | null = null;
  // Releases must authenticate as the account that took the lease, even after sign-out cleared the session.
  let leaseToken: string | null = null;

  const isStopped = (): boolean => disposed || owner.signal.aborted;

  const fireRelease = (): void => {
    void apiFetch(holdPath(leaseId), {
      headers: leaseToken ? { Authorization: `Bearer ${leaseToken}` } : undefined,
      keepalive: true,
      method: 'DELETE',
    }).catch(() => undefined);
    held = false;
    acknowledged = null;
  };

  /** Sorting and serialising up to 50k names per kind is the expensive part; an unchanged set reuses its body. */
  const bodyFor = (names: HeldMediaNames): string => {
    if (planned?.held === names) {
      return planned.body;
    }
    const imageSet = new Set(names.images);
    const videoSet = new Set(names.videos);
    if (planned && sameNames(imageSet, planned.images) && sameNames(videoSet, planned.videos)) {
      planned.held = names;
      return planned.body;
    }
    const body =
      imageSet.size === 0 && videoSet.size === 0
        ? ''
        : JSON.stringify({
            images: [...imageSet].sort().slice(0, MAX_HOLD_NAMES_PER_KIND),
            videos: [...videoSet].sort().slice(0, MAX_HOLD_NAMES_PER_KIND),
          });
    planned = { body, held: names, images: imageSet, videos: videoSet };
    return body;
  };

  const sendOnce = async (refresh: boolean): Promise<void> => {
    const body = bodyFor(read());
    if (isStopped()) {
      return;
    }
    if (!body) {
      if (held) {
        try {
          await apiFetch(holdPath(leaseId), { method: 'DELETE', signal: owner.signal });
          held = false;
          acknowledged = null;
        } catch {
          // Still held server-side; the next send or the expiry releases it.
        }
      }
      return;
    }
    if (!refresh && body === acknowledged) {
      return;
    }
    const token = getHttpAuthToken();
    try {
      await apiFetch(holdPath(leaseId), {
        body,
        headers: { 'Content-Type': 'application/json' },
        method: 'PUT',
        signal: owner.signal,
      });
    } catch {
      acknowledged = null;
      return;
    }
    leaseToken = token;
    if (disposed) {
      // Dispose already released what it knew about; this lease landed after.
      fireRelease();
      return;
    }
    held = true;
    acknowledged = body;
    lastSentAt = Date.now();
  };

  const send = async (refresh = false): Promise<void> => {
    if (isStopped()) {
      return;
    }
    if (inFlight) {
      pending = true;
      pendingRefresh ||= refresh;
      return;
    }
    inFlight = true;
    try {
      do {
        pending = false;
        const mustRefresh = refresh || pendingRefresh;
        refresh = false;
        pendingRefresh = false;
        await sendOnce(mustRefresh);
      } while (pending && !isStopped());
    } finally {
      inFlight = false;
    }
  };

  const schedule = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = null;
      void send();
    }, CHANGE_DEBOUNCE_MS);
  };
  const onVisibilityChange = (): void => {
    if (
      document.visibilityState === 'visible' &&
      (acknowledged === null || Date.now() - lastSentAt >= RESEND_ON_VISIBLE_AFTER_MS)
    ) {
      void send(true);
    }
  };
  const onPageHide = (): void => {
    if (held) {
      fireRelease();
    }
  };
  const onPageShow = (event: PageTransitionEvent): void => {
    if (event.persisted) {
      void send();
    }
  };

  const unsubscribe = subscribe(schedule);
  const heartbeat = setInterval(() => void send(true), HEARTBEAT_MS);
  document.addEventListener('visibilitychange', onVisibilityChange);
  window.addEventListener('pagehide', onPageHide);
  window.addEventListener('pageshow', onPageShow);
  void send();

  return () => {
    disposed = true;
    unsubscribe();
    clearInterval(heartbeat);
    if (timer !== null) {
      clearTimeout(timer);
    }
    document.removeEventListener('visibilitychange', onVisibilityChange);
    window.removeEventListener('pagehide', onPageHide);
    window.removeEventListener('pageshow', onPageShow);
    if (held) {
      fireRelease();
    }
  };
};

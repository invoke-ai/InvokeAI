import type { AccountScope } from '@platform/state/accountLifecycle';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';

/** Gathers a screen's failures into one probe. */
const GALLERY_IMAGE_PROBE_FIRST_DELAY_MS = 2_000;
/** 409 mid-move, or down restarting; the last delay repeats. */
const GALLERY_IMAGE_PROBE_BUSY_DELAYS_MS: readonly number[] = [15_000, 30_000];

interface ImageRecoveryProbe {
  /** Oldest first, each with its failed load's recovery revision (0: plain URL). */
  readonly failed: Map<string, number>;
  /** Since the last success; picks the next wait. */
  busyAnswers: number;
  readonly owner: AccountScope;
  timer: ReturnType<typeof setTimeout> | null;
  visibilityListener: (() => void) | null;
  url: string | null;
}

let probe: ImageRecoveryProbe | null = null;
let recoveryRevision = 0;
const recoveryListeners = new Set<() => void>();

/** Images that failed before the latest rise reload. */
export const getGalleryImageRecoveryRevision = (): number => recoveryRevision;

export const subscribeGalleryImageRecovery = (listener: () => void): (() => void) => {
  recoveryListeners.add(listener);
  return () => recoveryListeners.delete(listener);
};

export const getRecoveredGalleryImageUrl = (url: string, revision: number): string => {
  if (revision === 0) {
    return url;
  }

  const recoveredUrl = new URL(url, window.location.href);
  recoveredUrl.searchParams.set('gallery_image_recovery', String(revision));
  return recoveredUrl.toString();
};

const clearWait = (current: ImageRecoveryProbe): void => {
  if (current.timer !== null) {
    clearTimeout(current.timer);
    current.timer = null;
  }
  if (current.visibilityListener !== null) {
    document.removeEventListener('visibilitychange', current.visibilityListener);
    current.visibilityListener = null;
  }
};

const dropProbe = (current: ImageRecoveryProbe): void => {
  clearWait(current);
  if (probe === current) {
    probe = null;
  }
};

const schedule = (current: ImageRecoveryProbe, delayMs: number): void => {
  current.timer = setTimeout(() => {
    current.timer = null;
    // Paused while hidden; runs as soon as the tab is shown.
    if (document.visibilityState === 'hidden') {
      current.visibilityListener = () => {
        if (document.visibilityState !== 'hidden') {
          clearWait(current);
          void runProbe(current);
        }
      };
      document.addEventListener('visibilitychange', current.visibilityListener);
      return;
    }
    void runProbe(current);
  }, delayMs);
};

/** A move holds the files, or the server is restarting. */
const WAITING_STATUSES = new Set([409, 502, 503, 504]);

/** `other` will not change by waiting: missing, refused, or a page in place of the image. */
const readAnswer = async (url: string, signal: AbortSignal): Promise<'other' | 'served' | 'wait'> => {
  let response: Response;

  try {
    response = await fetch(url, { cache: 'no-store', signal });
  } catch {
    // No answer: a restarting server refuses connections. An owner abort lands here too; its probe is already dropped.
    return 'wait';
  }

  void response.body?.cancel().catch(() => undefined);
  if (WAITING_STATUSES.has(response.status)) {
    return 'wait';
  }
  // A proxy's fallback or sign-in page is not the image served again.
  return response.ok && response.headers.get('content-type')?.startsWith('image/') ? 'served' : 'other';
};

const runProbe = async (current: ImageRecoveryProbe): Promise<void> => {
  const [entry] = current.failed;

  if (entry === undefined) {
    return;
  }

  const [url, failedRevision] = entry;

  current.url = url;
  const answer = await readAnswer(url, current.owner.signal);
  current.url = null;

  if (probe !== current) {
    return;
  }

  if (answer === 'wait') {
    // Nothing announces a move's end to this tab; this failure was the server's, so the next success reloads it.
    const delays = GALLERY_IMAGE_PROBE_BUSY_DELAYS_MS;
    current.busyAnswers += 1;
    current.failed.set(url, 0);
    schedule(current, delays[Math.min(current.busyAnswers, delays.length) - 1] ?? 0);
    return;
  }

  // Failed again after the server served it: broken, not waiting.
  const isFreshFailure = failedRevision === 0 || failedRevision < recoveryRevision;

  if (answer === 'served' && isFreshFailure) {
    current.failed.clear();
    current.busyAnswers = 0;
    recoveryRevision += 1;
    recoveryListeners.forEach((listener) => listener());
    return;
  }

  current.failed.delete(url);
  if (current.failed.size > 0) {
    schedule(current, GALLERY_IMAGE_PROBE_FIRST_DELAY_MS);
  }
};

/**
 * An element cannot see why it failed, and during an image move the server answers 409 for minutes. One probe per
 * account asks with a request that sees the status; a success raises the revision so every failed image reloads once.
 */
export const reportGalleryImageFailure = (
  url: string,
  revision: number,
  owner: AccountScope = captureAccountScope()
): void => {
  if (!isAccountScopeCurrent(owner) || /^(?:blob|data):/i.test(url)) {
    return;
  }

  if (probe?.owner !== owner) {
    const created: ImageRecoveryProbe = {
      busyAnswers: 0,
      failed: new Map(),
      owner,
      timer: null,
      url: null,
      visibilityListener: null,
    };
    owner.signal.addEventListener('abort', () => dropProbe(created), { once: true });
    probe = created;
  }

  // Elements sharing a URL may carry different revisions; a fresh failure among them must still recover.
  const reported = probe.failed.get(url);
  probe.failed.set(url, reported === undefined ? revision : Math.min(reported, revision));

  if (probe.timer === null && probe.visibilityListener === null && probe.url === null) {
    schedule(probe, GALLERY_IMAGE_PROBE_FIRST_DELAY_MS);
  }
};

import type { AccountScope } from '@platform/state/accountLifecycle';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';

/** Gathers the failures a screen of images reports together into one probe. */
const GALLERY_IMAGE_PROBE_FIRST_DELAY_MS = 2_000;
/**
 * The waits between probes while the server is not serving images yet (409 during a move, or down while it
 * restarts); the last repeats for as long as that lasts.
 */
const GALLERY_IMAGE_PROBE_BUSY_DELAYS_MS: readonly number[] = [15_000, 30_000];

/** Probing for one account lifetime; dropped with it. */
interface ImageRecoveryProbe {
  /**
   * Failed image URLs waiting for the server to serve images again, oldest first, each with the recovery revision
   * its failed load carried (0 for the plain URL).
   */
  readonly failed: Map<string, number>;
  /** Waiting answers since the last success, which pick the next wait. */
  busyAnswers: number;
  readonly owner: AccountScope;
  timer: ReturnType<typeof setTimeout> | null;
  /** Set while a probe waits for the document to be shown. */
  visibilityListener: (() => void) | null;
  /** Set while a probe request is out. */
  url: string | null;
}

let probe: ImageRecoveryProbe | null = null;
let recoveryRevision = 0;
const recoveryListeners = new Set<() => void>();

/** Rises each time a probe sees the server serve a failed image again; images that failed before it reload. */
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
    // Paused while hidden: nobody sees the images, and the probe runs as soon as the tab is shown.
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

/** Statuses that mean the server cannot serve images yet: a move holds the files, or the server is restarting. */
const WAITING_STATUSES = new Set([409, 502, 503, 504]);

/**
 * The probe's answer: `wait` while the server cannot serve images yet, `served` once it serves the image again,
 * `other` for anything that will not change by waiting (missing, refused, or a page in place of the image).
 */
const readAnswer = async (url: string, signal: AbortSignal): Promise<'other' | 'served' | 'wait'> => {
  let response: Response;

  try {
    response = await fetch(url, { cache: 'no-store', signal });
  } catch {
    // No answer at all: a server restarting mid-move refuses connections until it is back. An abort from the owner
    // also lands here; its probe is already dropped and does not reschedule.
    return 'wait';
  }

  // Only the answer matters; the image itself is read again by the elements that failed.
  void response.body?.cancel().catch(() => undefined);
  if (WAITING_STATUSES.has(response.status)) {
    return 'wait';
  }
  // A page in place of the image (a proxy's fallback or sign-in page) is not the image served again.
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
    // A move or restart can last minutes and nothing announces its end to this tab. Whatever that image's earlier
    // reloads did, this failure was the server's, so the next success reloads it.
    const delays = GALLERY_IMAGE_PROBE_BUSY_DELAYS_MS;
    current.busyAnswers += 1;
    current.failed.set(url, 0);
    schedule(current, delays[Math.min(current.busyAnswers, delays.length) - 1] ?? 0);
    return;
  }

  // An image whose failed load already carried the current recovery revision failed after the server served it
  // again: it is broken, not waiting, and another reload would not help.
  const isFreshFailure = failedRevision === 0 || failedRevision < recoveryRevision;

  if (answer === 'served' && isFreshFailure) {
    current.failed.clear();
    current.busyAnswers = 0;
    recoveryRevision += 1;
    recoveryListeners.forEach((listener) => listener());
    return;
  }

  // Missing, refused for another reason, not an image, or served yet still failing after a reload: not waiting.
  current.failed.delete(url);
  if (current.failed.size > 0) {
    schedule(current, GALLERY_IMAGE_PROBE_FIRST_DELAY_MS);
  }
};

/**
 * An image element failed to load `url`, carrying recovery `revision` (0 for the plain URL). An element cannot see
 * why, and during an image move the server answers every image read with 409 for minutes. One probe per account
 * lifetime asks for one failed URL at a time with a request that can see the status: while the server cannot serve
 * images (409, 502-504, no connection) the probe keeps asking, paused while the tab is hidden; a success raises the
 * recovery revision so every image that failed before it reloads once; any other answer drops that URL. An account
 * change stops the probe and forgets its URLs.
 */
export const reportGalleryImageFailure = (
  url: string,
  revision: number,
  owner: AccountScope = captureAccountScope()
): void => {
  if (!isAccountScopeCurrent(owner) || /^(?:blob|data):/i.test(url)) {
    return;
  }

  // The previous lifetime's probe dropped itself when that lifetime ended.
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

import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { ApiError, HttpRequestIdentityExpiredError } from '@platform/transport/http';

import {
  applyServerOutputRead,
  EMPTY_TRANSCRIPT,
  TRANSCRIPT_LIMITS,
  type ServerOutputRead,
  type Transcript,
  type TranscriptLimits,
} from './transcript';

/**
 * `connecting` until the first read lands; `reconnecting` after a transient failure (held rows stay); `refused` (401,
 * 403) and `unavailable` (404) clear the rows and stop.
 */
export type ServerOutputRefusalReason = 'forbidden' | 'unavailable';

export type ServerOutputReaderPhase = 'connecting' | 'live' | 'reconnecting' | 'refused' | 'unavailable';

export interface ServerOutputReaderSnapshot {
  readonly phase: ServerOutputReaderPhase;
  readonly transcript: Transcript;
}

export interface ServerOutputReaderTiming {
  /** After a read that changed something. */
  readonly activeMs: number;
  /** After the first unchanged read; doubles per unchanged read up to `idleMaxMs`. */
  readonly idleMs: number;
  readonly idleMaxMs: number;
  /** After the first transient failure; doubles per failure up to `errorMaxMs`. */
  readonly errorMs: number;
  readonly errorMaxMs: number;
}

export const SERVER_OUTPUT_READER_TIMING: ServerOutputReaderTiming = {
  activeMs: 250,
  errorMaxMs: 15_000,
  errorMs: 1_000,
  idleMaxMs: 2_000,
  idleMs: 500,
};

export type ServerOutputReadPort = (cursor: string | null, signal: AbortSignal) => Promise<ServerOutputRead>;

export interface DocumentVisibility {
  isHidden(): boolean;
  subscribe(listener: () => void): () => void;
}

const documentVisibility: DocumentVisibility = {
  isHidden: () => document.visibilityState === 'hidden',
  subscribe: (listener) => {
    document.addEventListener('visibilitychange', listener);
    return () => document.removeEventListener('visibilitychange', listener);
  },
};

export interface ServerOutputReaderOptions {
  read: ServerOutputReadPort;
  /** Aborts when the account lifetime that owns this reader ends: rows clear and reading stops for good. */
  lifetime: AbortSignal;
  /**
   * Called once a read is refused (`forbidden`: 401 or 403) or capture is unavailable (404), so the caller can say why
   * and re-check the grant.
   */
  onRefused?: (reason: ServerOutputRefusalReason) => void;
  limits?: TranscriptLimits;
  timing?: ServerOutputReaderTiming;
  visibility?: DocumentVisibility;
}

export interface ServerOutputReader {
  getSnapshot(): ServerOutputReaderSnapshot;
  subscribe(listener: () => void): () => void;
  /** Reads while started and the document is visible. Returns the stop: in-flight reads abort, held rows stay. */
  start(): () => void;
}

const isAbort = (error: unknown): boolean =>
  (error instanceof DOMException && error.name === 'AbortError') || error instanceof HttpRequestIdentityExpiredError;

/**
 * Polls one transcript: at most one read in flight, the next scheduled after it settles. Every stop, hide and
 * restart takes a new generation, and a settlement from an older generation is dropped unapplied.
 */
export const createServerOutputReader = ({
  lifetime,
  limits = TRANSCRIPT_LIMITS,
  onRefused,
  read,
  timing = SERVER_OUTPUT_READER_TIMING,
  visibility = documentVisibility,
}: ServerOutputReaderOptions): ServerOutputReader => {
  const store = createExternalStoreCore<ServerOutputReaderSnapshot>({
    phase: 'connecting',
    transcript: EMPTY_TRANSCRIPT,
  });
  let cursor: string | null = null;
  let generation = 0;
  let controller: AbortController | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let started = false;
  let idleDelay = timing.idleMs;
  let errorDelay = timing.errorMs;

  const isStopped = (): boolean => {
    const { phase } = store.getSnapshot();
    return phase === 'refused' || phase === 'unavailable' || lifetime.aborted;
  };

  const halt = (): void => {
    generation += 1;
    controller?.abort();
    controller = null;
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const schedule = (delay: number): void => {
    timer = setTimeout(() => {
      timer = null;
      void pump();
    }, delay);
  };

  const finish = (phase: 'refused' | 'unavailable'): void => {
    halt();
    cursor = null;
    store.setSnapshot({ phase, transcript: EMPTY_TRANSCRIPT });
    onRefused?.(phase === 'refused' ? 'forbidden' : 'unavailable');
  };

  const pump = async (): Promise<void> => {
    if (!started || isStopped() || visibility.isHidden()) {
      return;
    }

    const owner = generation;
    const request = new AbortController();
    const sentCursor = cursor;
    controller = request;

    try {
      const result = await read(sentCursor, request.signal);

      if (owner !== generation) {
        return;
      }

      controller = null;
      const current = store.getSnapshot();
      const transcript = applyServerOutputRead(current.transcript, result, limits);
      const changed = transcript !== current.transcript || current.phase !== 'live';

      cursor = result.cursor;
      errorDelay = timing.errorMs;
      store.setSnapshot({ phase: 'live', transcript });

      if (result.has_more) {
        idleDelay = timing.idleMs;
        schedule(0);
      } else if (changed) {
        idleDelay = timing.idleMs;
        schedule(timing.activeMs);
      } else {
        schedule(idleDelay);
        idleDelay = Math.min(idleDelay * 2, timing.idleMaxMs);
      }
    } catch (error) {
      if (owner !== generation || isAbort(error) || lifetime.aborted) {
        return;
      }

      controller = null;

      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        finish('refused');
        return;
      }

      if (error instanceof ApiError && error.status === 404) {
        finish('unavailable');
        return;
      }

      // A cursor the server no longer accepts: start over from a fresh snapshot rather than stop.
      if (error instanceof ApiError && error.status === 422 && sentCursor !== null) {
        cursor = null;
        schedule(0);
        return;
      }

      store.patchSnapshot({ phase: 'reconnecting' });
      schedule(errorDelay);
      errorDelay = Math.min(errorDelay * 2, timing.errorMaxMs);
    }
  };

  const resume = (): void => {
    halt();
    idleDelay = timing.idleMs;
    void pump();
  };

  const onVisibilityChange = (): void => {
    if (visibility.isHidden()) {
      halt();
    } else {
      resume();
    }
  };

  const onLifetimeEnd = (): void => {
    halt();
    cursor = null;
    store.setSnapshot({ phase: 'connecting', transcript: EMPTY_TRANSCRIPT });
  };

  return {
    getSnapshot: store.getSnapshot,
    subscribe: store.subscribe,
    start: () => {
      if (started) {
        return () => {};
      }

      started = true;
      const unsubscribeVisibility = visibility.subscribe(onVisibilityChange);
      lifetime.addEventListener('abort', onLifetimeEnd);

      if (lifetime.aborted) {
        onLifetimeEnd();
      } else {
        resume();
      }

      return () => {
        started = false;
        unsubscribeVisibility();
        lifetime.removeEventListener('abort', onLifetimeEnd);
        halt();
      };
    },
  };
};

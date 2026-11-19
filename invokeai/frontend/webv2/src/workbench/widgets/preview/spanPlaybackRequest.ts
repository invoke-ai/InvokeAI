import type { GalleryItemKey } from '@features/gallery';

/** Tokenized trim-playback intents; stale ones expire to prevent unexpected delayed audio. */

export interface VideoSpanPlaybackRequest {
  endSeconds: number;
  itemKey: GalleryItemKey;
  requestedAt: number;
  startSeconds: number;
  token: number;
}

export const VIDEO_SPAN_PLAYBACK_TTL_MS = 15_000;

let currentRequest: VideoSpanPlaybackRequest | null = null;
let nextToken = 0;

const listeners = new Set<() => void>();

const notifyListeners = (): void => {
  for (const listener of listeners) {
    listener();
  }
};

/** Publishes a request and returns its token, the handle playback state is reported under. */
export const requestVideoSpanPlayback = (span: {
  endSeconds: number;
  itemKey: GalleryItemKey;
  startSeconds: number;
}): number => {
  nextToken += 1;
  currentRequest = { ...span, requestedAt: Date.now(), token: nextToken };
  notifyListeners();

  return nextToken;
};

export const getVideoSpanPlaybackRequest = (): VideoSpanPlaybackRequest | null => currentRequest;

export const isVideoSpanPlaybackFresh = (requestedAt: number): boolean =>
  Date.now() - requestedAt <= VIDEO_SPAN_PLAYBACK_TTL_MS;

/** Retires `token`, whether or not the player could act on it. A newer request is left alone. */
export const consumeVideoSpanPlaybackRequest = (token: number): void => {
  if (currentRequest?.token !== token) {
    return;
  }

  currentRequest = null;
  notifyListeners();
};

export const subscribeVideoSpanPlaybackRequests = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
};

/** Actual element playback of the current span, null when unused, left or hidden; pausing keeps it armed. */
export interface VideoSpanPlaybackState {
  isPlaying: boolean;
  pause: () => void;
  token: number;
}

let currentState: VideoSpanPlaybackState | null = null;

const stateListeners = new Set<() => void>();

const notifyStateListeners = (): void => {
  for (const listener of stateListeners) {
    listener();
  }
};

/** The player's report. Replaces whatever stood before: one span is on screen at a time. */
export const publishVideoSpanPlaybackState = (state: VideoSpanPlaybackState): void => {
  if (
    currentState !== null &&
    currentState.token === state.token &&
    currentState.isPlaying === state.isPlaying &&
    currentState.pause === state.pause
  ) {
    return;
  }

  currentState = state;
  notifyStateListeners();
};

/** Retires `token`'s state. A report from a newer span is left alone. */
export const clearVideoSpanPlaybackState = (token: number): void => {
  if (currentState?.token !== token) {
    return;
  }

  currentState = null;
  notifyStateListeners();
};

export const getVideoSpanPlaybackState = (): VideoSpanPlaybackState | null => currentState;

export const subscribeVideoSpanPlaybackState = (listener: () => void): (() => void) => {
  stateListeners.add(listener);

  return () => {
    stateListeners.delete(listener);
  };
};

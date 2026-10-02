import type { GalleryItemKey } from '@features/gallery';

/**
 * Publish ephemeral tokenized trim-loop intents after selecting media and raising Preview. Expire stale requests
 * to prevent unexpected delayed audio; the player reports running state and stop control under the same token.
 */

export interface VideoSpanPlaybackRequest {
  endSeconds: number;
  itemKey: GalleryItemKey;
  requestedAt: number;
  startSeconds: number;
  token: number;
}

/** How long a published span request stays honourable. */
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

/** True while `requestedAt` is recent enough to act on. */
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

/**
 * Report actual element playback for the current loop, null before use, after leaving its span, or while hidden.
 * Pausing keeps the loop armed for native resume.
 */
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

/** The player's report. Replaces whatever stood before: one loop is on screen at a time. */
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

/** Retires `token`'s state. A report from a newer loop is left alone. */
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

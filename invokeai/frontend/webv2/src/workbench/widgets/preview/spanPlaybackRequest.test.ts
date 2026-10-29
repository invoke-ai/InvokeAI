import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  clearVideoSpanPlaybackState,
  consumeVideoSpanPlaybackRequest,
  getVideoSpanPlaybackRequest,
  getVideoSpanPlaybackState,
  isVideoSpanPlaybackFresh,
  publishVideoSpanPlaybackState,
  requestVideoSpanPlayback,
  subscribeVideoSpanPlaybackRequests,
  subscribeVideoSpanPlaybackState,
  VIDEO_SPAN_PLAYBACK_TTL_MS,
} from './spanPlaybackRequest';

const retireOutstandingRequest = (): void => {
  const request = getVideoSpanPlaybackRequest();

  if (request) {
    consumeVideoSpanPlaybackRequest(request.token);
  }

  const state = getVideoSpanPlaybackState();

  if (state) {
    clearVideoSpanPlaybackState(state.token);
  }
};

afterEach(() => {
  retireOutstandingRequest();
  vi.useRealTimers();
});

describe('video span playback requests', () => {
  it('mints a fresh token per gesture, even on an unchanged span', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeVideoSpanPlaybackRequests(listener);
    const span = { endSeconds: 4, itemKey: 'video:clip.mp4', startSeconds: 2 } as const;

    requestVideoSpanPlayback(span);
    const first = getVideoSpanPlaybackRequest();

    requestVideoSpanPlayback(span);
    const second = getVideoSpanPlaybackRequest();

    expect(listener).toHaveBeenCalledTimes(2);
    expect(second).toMatchObject(span);
    // Pressing play again on a window the user has not touched must still read as a new
    // request: the player has already retired the first one.
    expect(second?.token).not.toBe(first?.token);

    unsubscribe();
    // The publisher gets the token back: it is what the player's report is matched on.
    expect(requestVideoSpanPlayback(span)).toBe(getVideoSpanPlaybackRequest()?.token);
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it('retires only the request the consumer actually read', () => {
    requestVideoSpanPlayback({ endSeconds: 4, itemKey: 'video:first.mp4', startSeconds: 2 });
    const stale = getVideoSpanPlaybackRequest();

    requestVideoSpanPlayback({ endSeconds: 9, itemKey: 'video:second.mp4', startSeconds: 8 });
    // A player that reads late must not clear the gesture that superseded it — pressing
    // play on a second reference while the first is still resolving is exactly that race.
    consumeVideoSpanPlaybackRequest(stale?.token ?? -1);

    expect(getVideoSpanPlaybackRequest()?.itemKey).toBe('video:second.mp4');

    consumeVideoSpanPlaybackRequest(getVideoSpanPlaybackRequest()?.token ?? -1);

    expect(getVideoSpanPlaybackRequest()).toBeNull();
  });

  it('stops vouching for a gesture the user has moved on from', () => {
    vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ['Date'] });
    requestVideoSpanPlayback({ endSeconds: 4, itemKey: 'video:clip.mp4', startSeconds: 2 });
    const request = getVideoSpanPlaybackRequest();

    expect(isVideoSpanPlaybackFresh(request?.requestedAt ?? 0)).toBe(true);

    vi.setSystemTime(Date.now() + VIDEO_SPAN_PLAYBACK_TTL_MS + 1);

    expect(isVideoSpanPlaybackFresh(request?.requestedAt ?? 0)).toBe(false);
  });
});

describe('video span playback state', () => {
  it('reports one loop at a time and tells subscribers only about changes', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeVideoSpanPlaybackState(listener);
    const pause = vi.fn();

    publishVideoSpanPlaybackState({ isPlaying: true, pause, token: 1 });
    expect(getVideoSpanPlaybackState()).toMatchObject({ isPlaying: true, token: 1 });
    expect(listener).toHaveBeenCalledTimes(1);

    // Deduplicate identical playback reports so repeated handlers do not rerender subscribers.
    publishVideoSpanPlaybackState({ isPlaying: true, pause, token: 1 });
    expect(listener).toHaveBeenCalledTimes(1);

    publishVideoSpanPlaybackState({ isPlaying: false, pause, token: 1 });
    expect(getVideoSpanPlaybackState()?.isPlaying).toBe(false);
    expect(listener).toHaveBeenCalledTimes(2);

    // A newer loop replaces the report outright: two cannot be on screen at once.
    publishVideoSpanPlaybackState({ isPlaying: true, pause, token: 2 });
    expect(getVideoSpanPlaybackState()?.token).toBe(2);

    unsubscribe();
    clearVideoSpanPlaybackState(2);
    expect(getVideoSpanPlaybackState()).toBeNull();
    expect(listener).toHaveBeenCalledTimes(3);
  });

  it('retires only the loop the caller was reporting on', () => {
    const pause = vi.fn();

    publishVideoSpanPlaybackState({ isPlaying: true, pause, token: 1 });
    publishVideoSpanPlaybackState({ isPlaying: true, pause, token: 2 });
    // A player unmounting late — the hidden keep-alive of a clip a newer request has
    // already replaced — must not take the live loop's report down with it.
    clearVideoSpanPlaybackState(1);

    expect(getVideoSpanPlaybackState()?.token).toBe(2);
  });
});

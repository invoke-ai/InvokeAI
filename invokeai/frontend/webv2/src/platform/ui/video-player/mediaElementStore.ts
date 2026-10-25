import { createExternalStoreCore } from '@platform/state/externalStoreCore';

import type { PlaybackWindow } from './videoTimeline';

/** What the attached media element reports; a detached store reads as an empty, paused element. */
export interface MediaElementSnapshot {
  currentTime: number;
  /** Finite seconds once known; 0 while unknown or unbounded. */
  duration: number;
  /** The element holds a media error; a new source clears it. */
  error: boolean;
  muted: boolean;
  paused: boolean;
  playbackRate: number;
  /** Metadata loaded: duration is known and seeks can land. */
  ready: boolean;
  volume: number;
  /** Playback is wanted but stalled on data or a seek. */
  waiting: boolean;
}

export interface MediaElementStore {
  /**
   * A React 19 ref callback: binds the store to `video` and returns the unbinding. Unbinding removes every listener
   * and frame callback and resets the snapshot; it leaves playback alone.
   */
  attach: (video: HTMLVideoElement) => () => void;
  getSnapshot: () => MediaElementSnapshot;
  pause: () => void;
  /** Playback the browser refuses (autoplay policy, a missing source) leaves the element paused, as reported. */
  play: () => void;
  /** Clamped to the media. */
  seek: (seconds: number) => void;
  /**
   * While set, playback that reaches the window's end wraps to its start with `loop`, including at the media end;
   * without it, playback pauses and rests at `restSeconds`. A seek is not playback: one that lands at or past the end
   * never wraps or pauses, though playback from before the window does once it reaches the end. Playing from the rest, or past it, starts the window over, whatever starts the
   * playback. Returns the clearing; a later window replaces this one, and clearing a replaced window does nothing.
   */
  setPlaybackWindow: (window: PlaybackWindow, options: { loop: boolean; restSeconds: number }) => () => void;
  setMuted: (muted: boolean) => void;
  setPlaybackRate: (rate: number) => void;
  /** Also unmutes for an audible volume. */
  setVolume: (volume: number) => void;
  subscribe: (listener: () => void) => () => void;
  togglePlay: () => void;
}

const DETACHED_SNAPSHOT: MediaElementSnapshot = {
  currentTime: 0,
  duration: 0,
  error: false,
  muted: false,
  paused: true,
  playbackRate: 1,
  ready: false,
  volume: 1,
  waiting: false,
};

const MEDIA_EVENTS = [
  'canplay',
  'durationchange',
  'emptied',
  'ended',
  'error',
  'loadeddata',
  'loadedmetadata',
  'loadstart',
  'pause',
  'play',
  'playing',
  'ratechange',
  'seeked',
  'seeking',
  'timeupdate',
  'volumechange',
  'waiting',
] as const;

const readSnapshot = (video: HTMLVideoElement): MediaElementSnapshot => ({
  currentTime: video.currentTime,
  duration: Number.isFinite(video.duration) ? video.duration : 0,
  error: video.error !== null,
  muted: video.muted,
  paused: video.paused,
  playbackRate: video.playbackRate,
  ready: video.readyState >= HTMLMediaElement.HAVE_METADATA,
  volume: video.volume,
  // An error can leave the element unpaused with no data coming; that is not waiting.
  waiting:
    !video.paused &&
    !video.ended &&
    video.error === null &&
    (video.seeking || video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA),
});

const seekElement = (video: HTMLVideoElement, seconds: number): void => {
  try {
    video.currentTime = seconds;
  } catch {
    // Some browsers throw while nothing is seekable yet; the element keeps its position.
  }
};

const ignoreRefusal = (): void => {};

// A playhead this close to a window's rest is at it: seeks land within a decoder's rounding of their target.
const REST_TOLERANCE_SECONDS = 0.001;

/**
 * One media element's state as an external store, for `useSyncExternalStore` with narrow selectors. Every media
 * event re-reads the element, so the snapshot cannot drift from it. The playhead follows animation frames only while
 * playing, for a smooth scrubber; `timeupdate` covers paused seeks and background tabs, where frames stop.
 */
export const createMediaElementStore = (): MediaElementStore => {
  const core = createExternalStoreCore<MediaElementSnapshot>(DETACHED_SNAPSHOT);
  let element: HTMLVideoElement | null = null;
  let frame: number | null = null;
  let playbackWindow: { loop: boolean; range: PlaybackWindow; restSeconds: number } | null = null;
  // The playhead at the last sample of uninterrupted playback; a seek clears it, so a landing is never a crossing.
  let sampledTime: number | null = null;

  const sync = (): void => core.patchSnapshot(element ? readSnapshot(element) : DETACHED_SNAPSHOT);
  const stopFrames = (): void => {
    if (frame !== null) {
      cancelAnimationFrame(frame);
      frame = null;
    }
  };
  const isAtRest = (video: HTMLVideoElement): boolean =>
    playbackWindow !== null &&
    !playbackWindow.loop &&
    (video.ended || video.currentTime >= playbackWindow.restSeconds - REST_TOLERANCE_SECONDS);
  // A finished window plays again from its start, as an ended clip does.
  const rewindFinishedWindow = (video: HTMLVideoElement): void => {
    if (playbackWindow && isAtRest(video)) {
      seekElement(video, playbackWindow.range.startSeconds);
      sampledTime = null;
    }
  };
  /**
   * Samples the playhead and acts when playback carried it to or past the window's end. Samples can be sparse
   * (throttled frames, timeupdate every quarter second), so playback that jumps from before the window to past it
   * between two samples has reached the end all the same. Frames and timeupdate both sample, so the window holds in
   * visible tabs to the frame and in background tabs at all; a window ending at the media end is reached when playback
   * ends, which pauses before any frame sees it.
   */
  const sampleWindowEnd = (): void => {
    if (!element) {
      return;
    }

    if (element.seeking) {
      sampledTime = null;
      return;
    }

    const previous = sampledTime;
    const current = element.currentTime;

    sampledTime = current;

    if (
      !playbackWindow ||
      previous === null ||
      (element.paused && !element.ended) ||
      previous >= playbackWindow.range.endSeconds ||
      current < playbackWindow.range.endSeconds
    ) {
      return;
    }

    sampledTime = null;

    if (playbackWindow.loop) {
      seekElement(element, playbackWindow.range.startSeconds);

      if (element.paused) {
        void element.play().catch(ignoreRefusal);
      }
    } else {
      element.pause();
      seekElement(element, playbackWindow.restSeconds);
    }
  };
  const tick = (): void => {
    frame = null;
    sampleWindowEnd();
    sync();

    if (element && !element.paused) {
      frame = requestAnimationFrame(tick);
    }
  };
  const startFrames = (): void => {
    if (frame === null && element && !element.paused) {
      frame = requestAnimationFrame(tick);
    }
  };
  const handleMediaEvent = (event: Event): void => {
    switch (event.type) {
      case 'seeking':
        sampledTime = null;
        break;
      case 'play':
        // Native play (media keys, a caller's own `play()`) replays a finished window as the controls do.
        if (element) {
          rewindFinishedWindow(element);
        }

        sampleWindowEnd();
        break;
      default:
        sampleWindowEnd();
    }

    if (element?.paused === false) {
      startFrames();
    } else {
      stopFrames();
    }

    sync();
  };
  const play = (): void => {
    if (!element) {
      return;
    }

    // Rewound before playing, so not even a frame plays from the rest.
    rewindFinishedWindow(element);
    void element.play().catch(ignoreRefusal);
  };

  return {
    attach: (video) => {
      element = video;

      for (const type of MEDIA_EVENTS) {
        video.addEventListener(type, handleMediaEvent);
      }

      startFrames();
      sync();

      return () => {
        for (const type of MEDIA_EVENTS) {
          video.removeEventListener(type, handleMediaEvent);
        }

        stopFrames();

        if (element === video) {
          element = null;
          sync();
        }
      };
    },
    getSnapshot: core.getSnapshot,
    pause: () => element?.pause(),
    play,
    seek: (seconds) => {
      if (!element) {
        return;
      }

      const duration = Number.isFinite(element.duration) ? element.duration : Number.POSITIVE_INFINITY;
      seekElement(element, Math.min(duration, Math.max(0, seconds)));
      // The element reports the target at once; its seeking event arrives a task later.
      sync();
    },
    setPlaybackWindow: (range, { loop, restSeconds }) => {
      const current = { loop, range, restSeconds };

      playbackWindow = current;

      return () => {
        if (playbackWindow === current) {
          playbackWindow = null;
        }
      };
    },
    setMuted: (muted) => {
      if (element) {
        element.muted = muted;
      }
    },
    setPlaybackRate: (rate) => {
      if (element) {
        element.playbackRate = rate;
      }
    },
    setVolume: (volume) => {
      if (!element) {
        return;
      }

      element.volume = Math.min(1, Math.max(0, volume));

      if (volume > 0) {
        element.muted = false;
      }
    },
    subscribe: core.subscribe,
    togglePlay: () => {
      if (!element) {
        return;
      }

      if (element.paused) {
        play();
      } else {
        element.pause();
      }
    },
  };
};

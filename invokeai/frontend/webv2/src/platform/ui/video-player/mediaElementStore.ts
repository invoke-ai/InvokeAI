import { createExternalStoreCore } from '@platform/state/externalStoreCore';

import type { PlaybackWindow } from './videoTimeline';

/** What the attached media element reports; a detached store reads as an empty, paused element. */
export interface MediaElementSnapshot {
  currentTime: number;
  /** Finite seconds once known; 0 while unknown or unbounded. */
  duration: number;
  error: boolean;
  muted: boolean;
  paused: boolean;
  playbackRate: number;
  ready: boolean;
  volume: number;
  waiting: boolean;
}

export interface MediaElementStore {
  /** Unbinding removes every listener and frame callback and resets the snapshot; playback is left alone. */
  attach: (video: HTMLVideoElement) => () => void;
  getSnapshot: () => MediaElementSnapshot;
  pause: () => void;
  /** Playback the browser refuses (autoplay policy, a missing source) leaves the element paused, as reported. */
  play: () => void;
  seek: (seconds: number) => void;
  /**
   * Playback reaching the window's end wraps with `loop`, else rests at `restSeconds`; seeks never wrap or pause.
   * Judged against the window set when playback starts; returns the clearing, a no-op once replaced.
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

/** Every media event re-reads the element; the playhead follows frames only while playing, `timeupdate` otherwise. */
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
  const replayFinishedWindow = (video: HTMLVideoElement): void => {
    if (
      playbackWindow &&
      !playbackWindow.loop &&
      video.currentTime >= playbackWindow.restSeconds - REST_TOLERANCE_SECONDS
    ) {
      seekElement(video, playbackWindow.range.startSeconds);
      sampledTime = null;
    }
  };
  /** Samples can be sparse (throttled frames, quarter-second timeupdate), so jumping past the end still reaches it. */
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
          replayFinishedWindow(element);
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
    replayFinishedWindow(element);
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

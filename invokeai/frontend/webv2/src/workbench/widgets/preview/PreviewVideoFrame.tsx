import type { GalleryItemKey } from '@features/gallery';
import type { VideoFrameCopyResult } from '@platform/ui/video-player/videoFrameCopyKeys';

import { getAuthSession, refreshProtectedMediaCookie } from '@features/identity';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useVideoErrorRecovery, VideoFailureOverlay } from '@platform/ui/video-player/VideoFailureOverlay';
import {
  copyVideoFrame,
  FIRST_FRAME_SEEK_SECONDS,
  isVideoFrameCopyAvailable,
  seekVideo,
} from '@platform/ui/video-player/videoFrameUtils';
import { isVideoPlayerControlsTarget, VideoPlayer } from '@platform/ui/video-player/VideoPlayer';
import { releaseFocusOnRemoval } from '@workbench/focusRegions';
import {
  useCallback,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type Ref,
  type RefObject,
  type SyntheticEvent,
} from 'react';

import type { PreviewMediaSource, PreviewVideoFrameController } from './PreviewFrame';

import {
  clearVideoSpanPlaybackState,
  getVideoSpanPlaybackRequest,
  isVideoSpanPlaybackFresh,
  publishVideoSpanPlaybackState,
  type VideoSpanPlaybackRequest,
} from './spanPlaybackRequest';

/** Audio settings live here for the page session (first clip at half volume); not account data. */
const previewVideoAudio = { muted: false, volume: 0.5 };

const rememberPreviewVideoAudio = (event: SyntheticEvent<HTMLVideoElement>): void => {
  previewVideoAudio.muted = event.currentTarget.muted;
  previewVideoAudio.volume = event.currentTarget.volume;
};

const getAccountEpoch = (): number => getAuthSession().accountEpoch;

/** A trimmed window of a clip, in seconds, as the Video panel's play buttons request it. */
interface VideoSpan {
  endSeconds: number;
  startSeconds: number;
}

/** Allows slight seek undershoot at span start; the player's own wraps must not retire the span. */
const SPAN_SEEK_TOLERANCE_SECONDS = 0.05;

/** How long a span waits for an element that could not act on it yet. */
const PARKED_SPAN_TTL_MS = 15_000;

/** Must stay a passive effect rendered after `VideoPlayer`, so it runs after the player binds its window. */
const SpanPlaybackStart = ({ onStart }: { onStart: () => void }) => {
  useMountEffect(() => {
    onStart();
  });

  return null;
};

export interface PreviewVideoPlayerProps {
  frameHeight: number;
  frameWidth: number;
  isControlsTargetRef: RefObject<((target: Element) => boolean) | null>;
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  loop?: boolean;
  onCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  onLoopChange?: (loop: boolean) => void;
  pendingSpanRequestRef: RefObject<VideoSpanPlaybackRequest | null>;
  playerSpanHandlerRef: RefObject<((request: VideoSpanPlaybackRequest) => void) | null>;
  source: Extract<PreviewMediaSource, { kind: 'video' }>;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
}

/** Lazy inner player loaded inside PreviewFrame's eager video shell on first video display. */
export const PreviewVideoPlayer = ({
  frameHeight,
  frameWidth,
  isControlsTargetRef,
  isItemCurrent,
  loop: loopPreference,
  onCopyAvailabilityChange,
  onLoopChange,
  pendingSpanRequestRef,
  playerSpanHandlerRef,
  source,
  videoControllerRef,
}: PreviewVideoPlayerProps) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const { handleError, hasFailed, retry } = useVideoErrorRecovery({
    getAccountEpoch,
    refreshCookie: refreshProtectedMediaCookie,
  });

  useMountEffect(() => {
    isControlsTargetRef.current = isVideoPlayerControlsTarget;
    return () => {
      isControlsTargetRef.current = null;
    };
  });

  const publishCopyAvailability = useCallback(() => {
    onCopyAvailabilityChange?.(source.itemKey, isVideoFrameCopyAvailable(videoRef.current));
  }, [onCopyAvailabilityChange, source.itemKey]);
  // The video can hold focus (its native controls, and the context menu key opens Preview's menu from it), so it hands
  // focus on when a later render removes it.
  const setVideoRef = useCallback(
    (video: HTMLVideoElement | null) => {
      if (!video) {
        return;
      }

      videoRef.current = video;
      video.muted = previewVideoAudio.muted;
      video.volume = previewVideoAudio.volume;
      onCopyAvailabilityChange?.(source.itemKey, isVideoFrameCopyAvailable(video));

      const releaseFocus = releaseFocusOnRemoval(video);

      return () => {
        videoRef.current = null;
        onCopyAvailabilityChange?.(source.itemKey, false);
        releaseFocus();
      };
    },
    [onCopyAvailabilityChange, source.itemKey]
  );
  const copyCurrentFrame = useCallback((): Promise<VideoFrameCopyResult> => {
    const video = videoRef.current;
    const accountEpoch = getAccountEpoch();
    const itemKey = source.itemKey;

    return copyVideoFrame(
      video,
      () =>
        videoRef.current === video && getAccountEpoch() === accountEpoch && (!isItemCurrent || isItemCurrent(itemKey))
    );
  }, [isItemCurrent, source.itemKey]);
  const isCopyAvailable = useCallback(() => isVideoFrameCopyAvailable(videoRef.current), []);
  // The span a request armed, which the player plays as its playback window.
  const [span, setSpan] = useState<VideoSpan | null>(null);
  const spanRef = useRef<VideoSpan | null>(null);
  // Expire spans parked before media readiness so retries or hidden remounts cannot start stale audio later.
  const parkedSpanRef = useRef<{ expiresAt: number; span: VideoSpan; token: number } | null>(null);
  // Keep request identity through pause/hide while the span stays armed; retire it when the playhead leaves or a newer
  // request arrives.
  const spanTokenRef = useRef<number | null>(null);
  // Where an armed span's playback starts once the player holds its window (see SpanPlaybackStart), until the same
  // deadline a parked span has: a start that a hide holds up still happens on show while the gesture is fresh, unless a
  // newer request has superseded it meanwhile.
  const pendingSpanStartRef = useRef<{ expiresAt: number; startSeconds: number; token: number } | null>(null);
  const [spanStartRequest, setSpanStartRequest] = useState(0);
  const armSpan = useCallback((next: VideoSpan | null) => {
    spanRef.current = next;
    setSpan(next);
  }, []);
  const pauseSpanPlayback = useCallback(() => {
    videoRef.current?.pause();
  }, []);
  // Publish actual element play/pause state, including the player's own controls and autoplay refusals.
  const publishSpanState = useCallback(() => {
    const token = spanTokenRef.current;
    const video = videoRef.current;

    if (token === null || !video) {
      return;
    }

    publishVideoSpanPlaybackState({ isPlaying: !video.paused, pause: pauseSpanPlayback, token });
  }, [pauseSpanPlayback]);
  const retireSpanToken = useCallback(() => {
    const token = spanTokenRef.current;

    spanTokenRef.current = null;

    if (token !== null) {
      clearVideoSpanPlaybackState(token);
    }
  }, []);
  const playSpan = useCallback(
    (request: VideoSpan, token: number, expiresAt?: number) => {
      const video = videoRef.current;
      // The deadline belongs to the gesture, not to this attempt: minting a fresh one on every re-park would let
      // hide/show cycles on a clip that never loads extend it indefinitely, which is the one thing the deadline exists
      // to stop.
      const deadline = expiresAt ?? Date.now() + PARKED_SPAN_TTL_MS;

      // Defer seeks until metadata and element attachment exist; resume from loadedmetadata or remount.
      if (!video || video.readyState < HTMLMediaElement.HAVE_METADATA) {
        parkedSpanRef.current = { expiresAt: deadline, span: request, token };
        return;
      }

      parkedSpanRef.current = null;
      const duration = Number.isFinite(video.duration) ? video.duration : null;
      const endSeconds = duration === null ? request.endSeconds : Math.min(request.endSeconds, duration);
      const startSeconds = Math.max(0, Math.min(request.startSeconds, endSeconds));
      // Arm any nonempty clamped span, including two-frame trims.
      const next = endSeconds > startSeconds ? { endSeconds, startSeconds } : null;

      // New requests replace prior span ownership; report only an armed span.
      retireSpanToken();
      armSpan(next);

      if (next !== null) {
        spanTokenRef.current = token;
      }

      pendingSpanStartRef.current = { expiresAt: deadline, startSeconds, token };
      setSpanStartRequest((current) => current + 1);
    },
    [armSpan, retireSpanToken]
  );
  const startPendingSpan = useCallback(() => {
    const pending = pendingSpanStartRef.current;
    const video = videoRef.current;

    pendingSpanStartRef.current = null;

    // On show this runs before the show effect consumes a request made while hidden; tokens only grow, so a pending
    // request newer than the held start supersedes it, and starting first would play the old span for a moment.
    const latestToken = Math.max(getVideoSpanPlaybackRequest()?.token ?? 0, pendingSpanRequestRef.current?.token ?? 0);

    if (!pending || !video || Date.now() > pending.expiresAt || latestToken > pending.token) {
      return;
    }

    seekVideo(video, pending.startSeconds);
    // Autoplay policy can refuse an unmuted play().
    void video.play().catch(() => {});
    publishSpanState();
  }, [pendingSpanRequestRef, publishSpanState]);
  // Retire spans only when seeks land outside them; own-seek markers fail for no-op or delayed seeks.
  const handleSeeking = useCallback(() => {
    const current = spanRef.current;
    const video = videoRef.current;

    if (!video) {
      return;
    }

    if (
      current &&
      video.currentTime >= current.startSeconds - SPAN_SEEK_TOLERANCE_SECONDS &&
      video.currentTime <= current.endSeconds
    ) {
      return;
    }

    parkedSpanRef.current = null;
    armSpan(null);
    retireSpanToken();
  }, [armSpan, retireSpanToken]);
  // Handle emptied because load() pauses without a pause event; otherwise the Video panel offers a stop that cannot
  // change anything.
  const handleEmptied = useCallback(() => {
    publishCopyAvailability();
    publishSpanState();
  }, [publishCopyAvailability, publishSpanState]);
  // Receive requests consumed by PreviewVideo's eager shell and resume parked spans on show.
  useMountEffect(() => {
    const handleConsumedRequest = (request: VideoSpanPlaybackRequest): void => {
      if (isVideoSpanPlaybackFresh(request.requestedAt)) {
        playSpan({ endSeconds: request.endSeconds, startSeconds: request.startSeconds }, request.token);
      }
    };

    playerSpanHandlerRef.current = handleConsumedRequest;

    const queuedRequest = pendingSpanRequestRef.current;
    pendingSpanRequestRef.current = null;

    if (queuedRequest) {
      parkedSpanRef.current = null;
      handleConsumedRequest(queuedRequest);
    } else {
      const parked = parkedSpanRef.current;

      if (parked) {
        parkedSpanRef.current = null;

        if (Date.now() <= parked.expiresAt) {
          playSpan(parked.span, parked.token, parked.expiresAt);
        }
      }
    }

    return () => {
      if (playerSpanHandlerRef.current === handleConsumedRequest) {
        playerSpanHandlerRef.current = null;
      }
      // Report nothing stoppable while hidden, but retain span identity for playback after showing.
      const token = spanTokenRef.current;

      if (token !== null) {
        clearVideoSpanPlaybackState(token);
      }
    };
  });
  useImperativeHandle(
    videoControllerRef,
    () => ({
      copyCurrentFrame,
      isCopyAvailable,
      itemKey: source.itemKey,
    }),
    [copyCurrentFrame, isCopyAvailable, source.itemKey]
  );
  const handleVideoError = useCallback(() => {
    publishCopyAvailability();
    const video = videoRef.current;
    handleError(video, () => videoRef.current === video && (!isItemCurrent || isItemCurrent(source.itemKey)));
  }, [handleError, isItemCurrent, publishCopyAvailability, source.itemKey]);
  // Seek slightly after metadata to clear the browser's show-poster flag and display native-resolution frame zero;
  // removing poster alone leaves black. This fetches extra media ranges. Skip when playback already began or the
  // playhead moved.
  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    const parked = parkedSpanRef.current;

    parkedSpanRef.current = null;

    // Honor only fresh parked spans; their requested seek also clears the poster flag.
    if (parked && Date.now() <= parked.expiresAt) {
      playSpan(parked.span, parked.token, parked.expiresAt);
      return;
    }

    if (!video) {
      return;
    }

    // After load() recovery, restore the armed span's position without starting audio or running a nudge that would
    // retire it.
    if (spanRef.current) {
      seekVideo(video, spanRef.current.startSeconds);
      return;
    }

    if (!video.paused || video.currentTime > 0) {
      return;
    }

    // Some browsers throw if the seekable range is not populated yet; the poster stays up.
    seekVideo(video, FIRST_FRAME_SEEK_SECONDS);
  }, [playSpan]);
  const handleRetry = useCallback(() => {
    retry(videoRef.current);
    publishCopyAvailability();
  }, [publishCopyAvailability, retry]);
  const failure = useMemo(
    () =>
      hasFailed ? (
        <VideoFailureOverlay
          height={frameHeight}
          poster={source.poster}
          size="full"
          width={frameWidth}
          onRetry={handleRetry}
        />
      ) : null,
    [frameHeight, frameWidth, handleRetry, hasFailed, source.poster]
  );

  return (
    <>
      <VideoPlayer
        ref={setVideoRef}
        aria-label={source.label}
        fps={source.fps}
        loop={loopPreference}
        overlay={failure}
        playbackWindow={span}
        poster={source.poster}
        size="full"
        src={source.src}
        onCanPlay={publishCopyAvailability}
        onEmptied={handleEmptied}
        // The player wraps a span ending on the clip's end in the same task as this; reporting here keeps the Video
        // panel's control from blinking to play on every pass.
        onEnded={publishSpanState}
        onError={handleVideoError}
        onLoadedData={publishCopyAvailability}
        onLoadedMetadata={handleLoadedMetadata}
        onLoopChange={onLoopChange}
        onPause={publishSpanState}
        onPlay={publishSpanState}
        onPlaying={publishCopyAvailability}
        onResize={publishCopyAvailability}
        onSeeked={publishCopyAvailability}
        onSeeking={handleSeeking}
        onVolumeChange={rememberPreviewVideoAudio}
        onWaiting={publishCopyAvailability}
      />
      {spanStartRequest > 0 ? <SpanPlaybackStart key={spanStartRequest} onStart={startPendingSpan} /> : null}
    </>
  );
};

export default PreviewVideoPlayer;

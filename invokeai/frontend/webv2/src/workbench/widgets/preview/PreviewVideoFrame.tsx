/* eslint-disable react/refs, react/immutability */
import type { GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type { VideoErrorRecovery } from '@platform/ui/video-player/videoFrameUtils';

import { Badge, Flex, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { getAuthSession, refreshProtectedMediaCookie } from '@features/identity';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import {
  copyVideoFrame,
  createVideoErrorRecovery,
  FIRST_FRAME_SEEK_SECONDS,
  isVideoFrameCopyAvailable,
  seekVideo,
} from '@platform/ui/video-player/videoFrameUtils';
import { isVideoPlayerControlsTarget, VideoPlayer } from '@platform/ui/video-player/VideoPlayer';
import { releaseFocusOnRemoval } from '@workbench/focusRegions';
import { GripHorizontalIcon } from 'lucide-react';
import {
  useCallback,
  useId,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent,
  type PointerEvent,
  type Ref,
  type SyntheticEvent,
} from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewMediaSource, PreviewVideoFrameController, PreviewVideoFrameCopyResult } from './PreviewFrame';

import { FittedFrame, PreviewStage } from './PreviewStage';
import { PreviewSwipeNeighbors } from './PreviewSwipeNeighbor';
import {
  clearVideoSpanPlaybackState,
  consumeVideoSpanPlaybackRequest,
  getVideoSpanPlaybackRequest,
  isVideoSpanPlaybackFresh,
  publishVideoSpanPlaybackState,
  subscribeVideoSpanPlaybackRequests,
} from './spanPlaybackRequest';
import { usePreviewSwipe, type PreviewSwipeNavigation } from './usePreviewSwipe';

/** Audio settings live here for the page session (first clip at half volume); not account data. */
const previewVideoAudio = { muted: false, volume: 0.5 };

const rememberPreviewVideoAudio = (event: SyntheticEvent<HTMLVideoElement>): void => {
  previewVideoAudio.muted = event.currentTarget.muted;
  previewVideoAudio.volume = event.currentTarget.volume;
};

/** A trimmed window of a clip, in seconds, as the Video panel's play buttons request it. */
interface VideoSpan {
  endSeconds: number;
  startSeconds: number;
}

/** Allows slight seek undershoot at span start; the player's own wraps must not retire the span. */
const SPAN_SEEK_TOLERANCE_SECONDS = 0.05;

/** How long a span waits for an element that could not act on it yet. */
const PARKED_SPAN_TTL_MS = 15_000;

const VIDEO_FAILURE_POSTER_STYLE: CSSProperties = {
  height: '100%',
  inset: 0,
  objectFit: 'contain',
  position: 'absolute',
  width: '100%',
};

/** Must stay a passive effect rendered after `VideoPlayer`, so it runs after the player binds its window. */
const SpanPlaybackStart = ({ onStart }: { onStart: () => void }) => {
  useMountEffect(() => {
    onStart();
  });

  return null;
};

export interface PreviewVideoFrameProps {
  dragItem?: GalleryItemRef;
  frameHeight: number;
  frameWidth: number;
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  loop?: boolean;
  onContextMenu?: (x: number, y: number) => void;
  onCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  onLoopChange?: (loop: boolean) => void;
  padding?: string;
  paddingBottom?: string;
  source: Extract<PreviewMediaSource, { kind: 'video' }>;
  swipe?: PreviewSwipeNavigation;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
}

export const PreviewVideoFrame = ({
  dragItem,
  frameHeight,
  frameWidth,
  isItemCurrent,
  loop: loopPreference,
  onContextMenu,
  onCopyAvailabilityChange,
  onLoopChange,
  padding,
  paddingBottom,
  source,
  swipe: swipeNavigation,
  videoControllerRef,
}: PreviewVideoFrameProps) => {
  const { t } = useTranslation();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // Use a grip for video so scrubbing the player cannot activate gallery dragging.
  const dragData = useMemo(() => (dragItem ? getGalleryItemDragData([dragItem]) : undefined), [dragItem]);
  const disabledDragId = useId();
  const dragId = dragItem
    ? getGalleryItemDragId(dragItem, 'preview-frame')
    : `preview-frame:disabled:${disabledDragId}`;
  const {
    isDragging,
    listeners,
    setNodeRef: setDragHandleRef,
  } = useDraggable({
    data: dragData,
    disabled: !dragItem,
    id: dragId,
  });
  // Each clip mounts its own player, so a landed swipe simply unmounts this one at rest.
  const swipe = usePreviewSwipe({
    displayedSourceToken: source.src,
    dragId,
    enabled: swipeNavigation !== undefined,
    isZoomed: false,
    navigation: swipeNavigation ?? null,
  });
  const { onPointerDown: handleSwipePointerDown } = swipe;
  // The control bar owns its touches: seeking and volume must never swipe.
  const handleStagePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const target = event.target;

      if (
        !(target instanceof Element) ||
        !event.currentTarget.contains(target) ||
        isVideoPlayerControlsTarget(target) ||
        target.ownerDocument.fullscreenElement?.contains(target)
      ) {
        return;
      }

      handleSwipePointerDown(event);
    },
    [handleSwipePointerDown]
  );
  const [hasFailed, setHasFailed] = useState(false);
  const recoveryRef = useRef<VideoErrorRecovery | null>(null);
  const getRecovery = useCallback(() => {
    if (!recoveryRef.current) {
      recoveryRef.current = createVideoErrorRecovery({
        getAccountEpoch: () => getAuthSession().accountEpoch,
        onFailedChange: setHasFailed,
        refreshCookie: refreshProtectedMediaCookie,
      });
    }
    return recoveryRef.current;
  }, []);
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
  const copyCurrentFrame = useCallback((): Promise<PreviewVideoFrameCopyResult> => {
    const video = videoRef.current;
    const accountEpoch = getAuthSession().accountEpoch;
    const itemKey = source.itemKey;

    return copyVideoFrame(
      video,
      () =>
        videoRef.current === video &&
        getAuthSession().accountEpoch === accountEpoch &&
        (!isItemCurrent || isItemCurrent(itemKey))
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
    const request = getVideoSpanPlaybackRequest();

    if (!pending || !video || Date.now() > pending.expiresAt || (request && request.token > pending.token)) {
      return;
    }

    seekVideo(video, pending.startSeconds);
    // Autoplay policy can refuse an unmuted play().
    void video.play().catch(() => {});
    publishSpanState();
  }, [publishSpanState]);
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
  // Subscribe by fixed item key and consume pending requests on show. The player pauses itself on hide and unmount.
  useMountEffect(() => {
    const itemKey = source.itemKey;
    const consumeSpanRequest = (): void => {
      const request = getVideoSpanPlaybackRequest();

      if (!request || request.itemKey !== itemKey) {
        return;
      }

      // Retired whether or not it is still honourable, so a request the user has moved on from cannot wait around for
      // the next player that shows this clip.
      consumeVideoSpanPlaybackRequest(request.token);

      if (isVideoSpanPlaybackFresh(request.requestedAt)) {
        playSpan({ endSeconds: request.endSeconds, startSeconds: request.startSeconds }, request.token);
      }
    };
    // Resume fresh parked spans on show if metadata finished while the ref was detached.
    const parked = parkedSpanRef.current;

    if (parked) {
      parkedSpanRef.current = null;

      if (Date.now() <= parked.expiresAt) {
        playSpan(parked.span, parked.token, parked.expiresAt);
      }
    }

    consumeSpanRequest();

    const unsubscribe = subscribeVideoSpanPlaybackRequests(consumeSpanRequest);

    return () => {
      unsubscribe();
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
    getRecovery().handleError(
      video,
      () => videoRef.current === video && (!isItemCurrent || isItemCurrent(source.itemKey))
    );
  }, [getRecovery, isItemCurrent, publishCopyAvailability, source.itemKey]);
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
    getRecovery().retry(videoRef.current);
    publishCopyAvailability();
  }, [getRecovery, publishCopyAvailability]);
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (onContextMenu) {
        event.preventDefault();
        onContextMenu(event.clientX, event.clientY);
      }
    },
    [onContextMenu]
  );
  const failure = hasFailed ? (
    <>
      <img
        aria-hidden="true"
        alt=""
        draggable={false}
        height={frameHeight}
        src={source.poster}
        style={VIDEO_FAILURE_POSTER_STYLE}
        width={frameWidth}
      />
      <Flex
        align="center"
        bg="blackAlpha.700"
        direction="column"
        gap="3"
        inset="0"
        justify="center"
        position="absolute"
      >
        <Text color="white" fontSize="lg" fontWeight="semibold">
          {t('widgets.preview.videoFailed')}
        </Text>
        <Button aria-label={t('widgets.preview.videoRetry')} size="lg" onClick={handleRetry}>
          {t('widgets.preview.videoRetry')}
        </Button>
      </Flex>
    </>
  ) : null;

  return (
    <PreviewStage
      ref={swipe.stageRefCallback}
      fill="flex"
      padding={padding}
      paddingBottom={paddingBottom}
      // One-finger travel is the swipe's, exactly as on the image stage: any native pan left available here lets the
      // browser (or the player's own controls) claim a swipe midway and cancel it. Without a swipe the browser keeps
      // its native gestures.
      touchAction={swipeNavigation ? 'none' : undefined}
      onPointerDown={swipeNavigation ? handleStagePointerDown : undefined}
    >
      <FittedFrame
        ref={swipe.contentTrackRef}
        bg="black"
        frameHeight={frameHeight}
        frameWidth={frameWidth}
        opacity={isDragging ? 0.55 : undefined}
        style={swipe.restStyle}
        onContextMenu={onContextMenu ? handleContextMenu : undefined}
      >
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
        {dragItem ? (
          // Keep drag grips unfocusable so KeyboardSensor cannot start invisible Enter/Space drags.
          <Badge
            ref={setDragHandleRef}
            {...listeners}
            cursor={isDragging ? 'grabbing' : 'grab'}
            // Top-center, clear of the control bar along the bottom edge.
            insetInlineStart="50%"
            position="absolute"
            title={t('widgets.preview.dragVideo')}
            top="2"
            touchAction="none"
            transform="translateX(-50%)"
            variant="solid"
          >
            <GripHorizontalIcon aria-hidden="true" size={12} />
          </Badge>
        ) : null}
      </FittedFrame>
      {swipeNavigation && swipe.showsNeighbors ? (
        <PreviewSwipeNeighbors
          neighbors={swipeNavigation.neighbors}
          nextTrackRef={swipe.nextTrackRef}
          previousTrackRef={swipe.previousTrackRef}
          restStyle={swipe.restStyle}
        />
      ) : null}
    </PreviewStage>
  );
};

export default PreviewVideoFrame;

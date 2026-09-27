/* eslint-disable react/refs, react/immutability */
import type { GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';

import { Badge, Flex, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { getAuthSession, refreshProtectedMediaCookie } from '@features/identity';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { GripHorizontalIcon } from 'lucide-react';
import {
  useCallback,
  useEffect,
  useId,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
  type Ref,
  type SyntheticEvent,
} from 'react';
import { useTranslation } from 'react-i18next';

import { PreviewCompareDropZone } from './PreviewCompareDropZone';
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
import { usePreviewLoupe, type PreviewLoupeControls, type PreviewZoomState } from './usePreviewLoupe';
import { usePreviewSwipe, type PreviewSwipeNavigation } from './usePreviewSwipe';

export type PreviewMediaSource =
  | { itemKey: GalleryItemKey; kind: 'image'; source: StreamingImageSource }
  | { itemKey: GalleryItemKey; kind: 'video'; label: string; poster: string; src: string };

interface PreviewFrameProps {
  children?: ReactNode;
  dragItem?: GalleryItemRef;
  frameHeight: number;
  frameWidth: number;
  /** Keep the last denoise frame until source decoding completes, then notify onSourceLoaded. */
  holdSource?: StreamingImageSource | null;
  onSourceLoaded?: (src: string) => void;
  /** A small, usually cached rendition (the thumbnail) shown until a settled image's full pixels load. */
  placeholderSrc?: string;
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  /** Keep live-frame geometry identical to finished media; progress/device text belongs to external chrome. */
  isLive: boolean;
  loupeControlsRef?: Ref<PreviewLoupeControls>;
  onContextMenu?: (x: number, y: number) => void;
  onZoomChange?: (state: PreviewZoomState) => void;
  onVideoCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  padding?: string;
  paddingBottom?: string;
  shouldAntialiasLiveImage: boolean;
  source: PreviewMediaSource | null;
  /** Touch swipe navigation between neighbors; framed, settled images only. */
  swipe?: PreviewSwipeNavigation;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
  variant: 'framed' | 'inset';
}

export const PreviewFrame = (props: PreviewFrameProps) => {
  if (props.source?.kind === 'video') {
    return (
      <PreviewVideo
        key={props.source.itemKey}
        dragItem={props.dragItem}
        frameHeight={props.frameHeight}
        frameWidth={props.frameWidth}
        isItemCurrent={props.isItemCurrent}
        onContextMenu={props.onContextMenu}
        onCopyAvailabilityChange={props.onVideoCopyAvailabilityChange}
        padding={props.padding}
        paddingBottom={props.paddingBottom}
        source={props.source}
        swipe={props.swipe}
        videoControllerRef={props.videoControllerRef}
      />
    );
  }

  return <PreviewImageFrame {...props} source={props.source?.source ?? null} />;
};

const PreviewImageFrame = ({
  children,
  dragItem,
  frameHeight,
  frameWidth,
  holdSource,
  isLive,
  loupeControlsRef,
  onContextMenu,
  onSourceLoaded,
  onZoomChange,
  padding,
  paddingBottom,
  placeholderSrc,
  shouldAntialiasLiveImage,
  source,
  swipe: swipeNavigation,
  variant,
}: Omit<PreviewFrameProps, 'isItemCurrent' | 'onVideoCopyAvailabilityChange' | 'source' | 'videoControllerRef'> & {
  source: StreamingImageSource | null;
}) => {
  const loupe = usePreviewLoupe({
    controlsRef: loupeControlsRef,
    enabled: variant === 'framed' && !isLive,
    naturalWidth: frameWidth,
    onZoomChange,
  });
  const { contentRefCallback, stageRefCallback } = loupe;
  const dragData = useMemo(() => (dragItem ? getGalleryItemDragData([dragItem]) : undefined), [dragItem]);
  const isDragDisabled = !dragItem || isLive || loupe.isZoomed;
  const disabledDragId = useId();
  const dragId = dragItem
    ? getGalleryItemDragId(dragItem, 'preview-frame')
    : `preview-frame:disabled:${disabledDragId}`;
  const {
    isDragging,
    listeners,
    setNodeRef: setDragNodeRef,
  } = useDraggable({
    data: dragData,
    disabled: isDragDisabled,
    id: dragId,
  });
  const isSwipeEnabled = variant === 'framed' && !isLive && swipeNavigation !== undefined;
  const displayedSourceToken = variant === 'framed' && !isLive && source ? source.src : null;
  const swipe = usePreviewSwipe({
    displayedSourceToken,
    dragId,
    enabled: isSwipeEnabled,
    isZoomed: loupe.isZoomed,
    navigation: swipeNavigation ?? null,
  });
  const { contentTrackRef, onPointerDown: handleSwipePointerDown, stageRefCallback: swipeStageRefCallback } = swipe;
  const setContentRef = useCallback(
    (element: HTMLDivElement | null) => {
      setDragNodeRef(element);
      contentTrackRef(element);
      const cleanupLoupe = contentRefCallback?.(element);

      return () => {
        cleanupLoupe?.();
        contentTrackRef(null);
        setDragNodeRef(null);
      };
    },
    [contentRefCallback, contentTrackRef, setDragNodeRef]
  );
  const setStageRef = useCallback(
    (element: HTMLDivElement | null) => {
      const cleanupLoupe = stageRefCallback?.(element);
      const cleanupSwipe = swipeStageRefCallback(element);

      return () => {
        cleanupLoupe?.();
        cleanupSwipe?.();
      };
    },
    [stageRefCallback, swipeStageRefCallback]
  );
  // Reset zoom in place when the displayed image changes (or goes live) — a
  // remount would flash the frame on every selection.
  loupe.syncDisplayedSource(displayedSourceToken);
  const loupeStageProps = loupe.stageProps;
  const handleStagePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      loupeStageProps?.onPointerDown(event);
      handleSwipePointerDown(event);
    },
    [handleSwipePointerDown, loupeStageProps]
  );
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (onContextMenu) {
        event.preventDefault();
        onContextMenu(event.clientX, event.clientY);
      }
    },
    [onContextMenu]
  );
  // `image-rendering` stays unset while the loupe is enabled so the loupe can
  // toggle pixelated rendering on the content box (it inherits to the img).
  const imageStyle = useMemo<CSSProperties>(
    () => ({
      display: 'block',
      height: 'auto',
      imageRendering: isLive && !shouldAntialiasLiveImage ? 'pixelated' : undefined,
      // Positioned, so it paints over the absolutely positioned placeholder beneath it.
      position: 'relative',
      width: '100%',
    }),
    [isLive, shouldAntialiasLiveImage]
  );
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [settledSrc, setSettledSrc] = useState<string | null>(null);
  const isHolding = Boolean(holdSource && source && settledSrc !== source.src);
  // Settled images mount one element per source: a reused element whose new source is still downloading would keep
  // painting the previous image until the new one arrived (a swipe landing on the wrong picture, then a flash). Live
  // frames keep reusing theirs, since each denoise step is a new data URL.
  const imageKey = isLive ? undefined : source?.src;
  const [loadedSrc, setLoadedSrc] = useState<string | null>(null);
  const isShowingPlaceholder = Boolean(placeholderSrc && !isLive && !isHolding && source && loadedSrc !== source.src);
  const handleSourceSettled = useCallback(
    (event: SyntheticEvent<HTMLImageElement>) => {
      const src = event.currentTarget.getAttribute('src');

      // Only a frame with a placeholder tracks loading; live frames never have one.
      if (src !== null && placeholderSrc) {
        setLoadedSrc(src);
      }

      // Only tracked while a hold is up: a live frame is a new data URL every
      // step, and settling each one would re-render the frame per step for
      // nothing.
      if (src === null || !holdSource) {
        return;
      }

      setSettledSrc(src);
      onSourceLoaded?.(src);
    },
    [holdSource, onSourceLoaded, placeholderSrc]
  );
  // A hold arriving after the image already decoded (a cached image, or the
  // element reused across a source swap) would never see a load event.
  useEffect(() => {
    const image = imageRef.current;

    if (
      !holdSource ||
      !source ||
      !image ||
      image.getAttribute('src') !== source.src ||
      !image.complete ||
      image.naturalWidth === 0
    ) {
      return;
    }

    setSettledSrc(source.src);
    onSourceLoaded?.(source.src);
  }, [holdSource, onSourceLoaded, source]);
  const heldImageStyle = useMemo<CSSProperties>(() => ({ ...imageStyle, visibility: 'hidden' }), [imageStyle]);
  const holdImageStyle = useMemo<CSSProperties>(
    () => ({
      height: '100%',
      imageRendering: shouldAntialiasLiveImage ? undefined : 'pixelated',
      inset: 0,
      objectFit: 'contain',
      pointerEvents: 'none',
      position: 'absolute',
      width: '100%',
    }),
    [shouldAntialiasLiveImage]
  );
  const media = source ? (
    <>
      {isShowingPlaceholder ? (
        // Keyed like the image above it, for the same reason: a reused element would show the last thumbnail.
        <img
          key={placeholderSrc}
          aria-hidden="true"
          alt=""
          draggable={false}
          src={placeholderSrc}
          style={PLACEHOLDER_IMAGE_STYLE}
        />
      ) : null}
      <img
        key={imageKey}
        ref={imageRef}
        alt={source.alt}
        draggable={false}
        height={frameHeight}
        src={source.src}
        style={isHolding ? heldImageStyle : imageStyle}
        width={frameWidth}
        onError={handleSourceSettled}
        onLoad={handleSourceSettled}
      />
      {/* The hidden finished image keeps the frame's geometry; the held frame
          only paints over it until the real pixels are ready. */}
      {isHolding && holdSource ? (
        <img aria-hidden="true" alt="" draggable={false} src={holdSource.src} style={holdImageStyle} />
      ) : null}
    </>
  ) : null;
  if (variant === 'inset') {
    return (
      // Reserve chrome inset consistently across live tiles; their dot grids still fill each cell.
      <PreviewStage fill="parent">
        {source ? (
          <FittedFrame frameHeight={frameHeight} frameWidth={frameWidth}>
            {media}
          </FittedFrame>
        ) : (
          children
        )}
      </PreviewStage>
    );
  }

  return (
    <PreviewStage
      ref={setStageRef}
      cursor={loupe.isZoomed ? 'grab' : undefined}
      fill="flex"
      padding={padding}
      paddingBottom={paddingBottom}
      // Suppress browser gestures across the loupe's entire stage; live renders without a loupe retain native
      // behavior.
      touchAction={isLive ? undefined : 'none'}
      {...loupe.stageProps}
      onPointerDown={handleStagePointerDown}
    >
      {/*
       * Disable comparison drops over live renders because comparison would pause follow and expose stale
       * selection.
       */}
      {isLive ? null : <PreviewCompareDropZone currentImageName={dragItem?.kind === 'image' ? dragItem.name : null} />}
      <FittedFrame
        ref={setContentRef}
        {...listeners}
        bg="transparent"
        cursor={isDragDisabled ? undefined : isDragging ? 'grabbing' : 'grab'}
        // One-finger movement swipes, so a touch drag needs the hold first (see holdToDragSensor).
        data-drag-hold-on-touch={isSwipeEnabled ? 'true' : undefined}
        frameHeight={frameHeight}
        frameWidth={frameWidth}
        opacity={isDragging ? 0.55 : undefined}
        style={swipe.restStyle}
        touchAction={isDragDisabled ? undefined : 'none'}
        onContextMenu={onContextMenu ? handleContextMenu : undefined}
      >
        {media}
      </FittedFrame>
      {isSwipeEnabled && swipe.showsNeighbors ? (
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

const PreviewVideo = ({
  dragItem,
  frameHeight,
  frameWidth,
  isItemCurrent,
  onContextMenu,
  onCopyAvailabilityChange,
  padding,
  paddingBottom,
  source,
  swipe: swipeNavigation,
  videoControllerRef,
}: {
  dragItem?: GalleryItemRef;
  frameHeight: number;
  frameWidth: number;
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  onContextMenu?: (x: number, y: number) => void;
  onCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  padding?: string;
  paddingBottom?: string;
  source: Extract<PreviewMediaSource, { kind: 'video' }>;
  swipe?: PreviewSwipeNavigation;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
}) => {
  const { t } = useTranslation();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // Use a corner drag grip for video so native seek-bar scrubbing cannot activate gallery dragging.
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
  // The native control bar owns touches along the frame's bottom edge: seeking and volume must never swipe. A
  // fullscreen player is out of the carousel entirely; navigating would unmount it and drop fullscreen.
  const handleStagePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const video = videoRef.current;
      const frame = video?.getBoundingClientRect();

      if (
        (frame && event.clientY >= frame.bottom - VIDEO_CONTROL_BAND_PX) ||
        (video && video.ownerDocument.fullscreenElement === video)
      ) {
        return;
      }

      handleSwipePointerDown(event);
    },
    [handleSwipePointerDown]
  );
  const automaticRefreshUsedRef = useRef(false);
  const pendingRefreshRef = useRef<Promise<boolean> | null>(null);
  const [hasFailed, setHasFailed] = useState(false);
  const publishCopyAvailability = useCallback(() => {
    onCopyAvailabilityChange?.(source.itemKey, isVideoFrameCopyAvailable(videoRef.current));
  }, [onCopyAvailabilityChange, source.itemKey]);
  const setVideoRef = useCallback(
    (video: HTMLVideoElement | null) => {
      videoRef.current = video;
      onCopyAvailabilityChange?.(source.itemKey, isVideoFrameCopyAvailable(video));
    },
    [onCopyAvailabilityChange, source.itemKey]
  );
  const copyCurrentFrame = useCallback(async (): Promise<PreviewVideoFrameCopyResult> => {
    const video = videoRef.current;
    const clipboardItem = globalThis.ClipboardItem;
    const write = navigator.clipboard?.write?.bind(navigator.clipboard);

    if (!clipboardItem || !write) {
      return { ok: false, reason: 'unsupported' };
    }

    if (!isVideoFrameReady(video)) {
      return { ok: false, reason: 'not-ready' };
    }

    const accountEpoch = getAuthSession().accountEpoch;
    const itemKey = source.itemKey;
    const width = video.videoWidth;
    const height = video.videoHeight;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');

    if (!context) {
      return { ok: false, reason: 'draw-failed' };
    }

    try {
      context.drawImage(video, 0, 0, width, height);
    } catch {
      return { ok: false, reason: 'draw-failed' };
    }

    let blob: Blob | null;

    try {
      blob = await encodeCanvasPng(canvas);
    } catch (error: unknown) {
      return { ok: false, reason: isCanvasSecurityError(error) ? 'draw-failed' : 'encode-failed' };
    }

    if (!blob) {
      return { ok: false, reason: 'encode-failed' };
    }

    const isCurrent = (): boolean =>
      videoRef.current === video &&
      getAuthSession().accountEpoch === accountEpoch &&
      (!isItemCurrent || isItemCurrent(itemKey));

    if (!isCurrent()) {
      return { ok: false, reason: 'stale' };
    }

    let item: ClipboardItem;

    try {
      item = new clipboardItem({ 'image/png': blob });
      await write([item]);
    } catch {
      return { ok: false, reason: 'clipboard-failed' };
    }

    return isCurrent() ? { ok: true } : { ok: false, reason: 'stale' };
  }, [isItemCurrent, source.itemKey]);
  const isCopyAvailable = useCallback(() => isVideoFrameCopyAvailable(videoRef.current), []);
  // Loop requested trim spans for preview without generation; keep loop state in refs because wraps change no
  // rendered UI.
  const spanRef = useRef<VideoSpan | null>(null);
  // Expire spans parked before media readiness so retries or hidden remounts cannot start stale audio later.
  const parkedSpanRef = useRef<{ expiresAt: number; span: VideoSpan; token: number } | null>(null);
  const spanFrameRef = useRef<number | null>(null);
  // Keep request identity through pause/hide while the loop stays armed; retire it when the playhead leaves or a
  // newer request arrives.
  const spanTokenRef = useRef<number | null>(null);
  const seekWithinSpan = useCallback((video: HTMLVideoElement, time: number) => {
    try {
      video.currentTime = time;
    } catch {
      // Some browsers throw while the seekable range is still empty; the watch retries on
      // its next tick.
    }
  }, []);
  const stopSpanWatch = useCallback(() => {
    if (spanFrameRef.current !== null) {
      cancelAnimationFrame(spanFrameRef.current);
      spanFrameRef.current = null;
    }
  }, []);
  const pauseSpanPlayback = useCallback(() => {
    videoRef.current?.pause();
  }, []);
  // Publish actual element play/pause state, including native controls and autoplay refusals.
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
  // Use rAF for precise visible-tab wrapping and timeupdate as the background-tab fallback.
  const enforceSpan = useCallback(() => {
    const span = spanRef.current;
    const video = videoRef.current;

    if (!span || !video || video.paused || video.currentTime < span.endSeconds) {
      return;
    }

    seekWithinSpan(video, span.startSeconds);
  }, [seekWithinSpan]);
  const startSpanWatch = useCallback(() => {
    if (spanRef.current === null || spanFrameRef.current !== null) {
      return;
    }

    const tick = (): void => {
      spanFrameRef.current = null;
      enforceSpan();

      if (spanRef.current !== null && videoRef.current?.paused === false) {
        spanFrameRef.current = requestAnimationFrame(tick);
      }
    };

    spanFrameRef.current = requestAnimationFrame(tick);
  }, [enforceSpan]);
  const playSpan = useCallback(
    (span: VideoSpan, token: number, expiresAt?: number) => {
      const video = videoRef.current;

      // Defer seeks until metadata and element attachment exist; resume from loadedmetadata or remount.
      if (!video || video.readyState < HTMLMediaElement.HAVE_METADATA) {
        // The deadline belongs to the gesture, not to this attempt: minting a fresh one on
        // every re-park would let hide/show cycles on a clip that never loads extend it
        // indefinitely, which is the one thing the deadline exists to stop.
        parkedSpanRef.current = { expiresAt: expiresAt ?? Date.now() + PARKED_SPAN_TTL_MS, span, token };
        return;
      }

      parkedSpanRef.current = null;
      const duration = Number.isFinite(video.duration) ? video.duration : null;
      const endSeconds = duration === null ? span.endSeconds : Math.min(span.endSeconds, duration);
      const startSeconds = Math.max(0, Math.min(span.startSeconds, endSeconds));

      // Loop any nonempty clamped span, including two-frame trims; collapsed/inverted bounds play without looping.
      spanRef.current = endSeconds > startSeconds ? { endSeconds, startSeconds } : null;
      // New requests replace prior loop ownership; report only an armed loop, not collapsed one-shot playback.
      retireSpanToken();

      if (spanRef.current !== null) {
        spanTokenRef.current = token;
      }

      seekWithinSpan(video, startSeconds);
      // Autoplay policy can refuse an unmuted play(). That leaves the clip parked on the
      // first selected frame with the native controls live and the loop still armed, so
      // pressing play there plays the selection rather than the whole clip.
      void video.play().catch(() => {});
      // Arm onPlay for asynchronous play(), and immediately for already-playing elements that emit no new play
      // event.
      startSpanWatch();
      publishSpanState();
    },
    [publishSpanState, retireSpanToken, seekWithinSpan, startSpanWatch]
  );
  // Retire loops only when seeks land outside the span; own-seek markers fail for no-op or delayed seeks.
  // In-window scrubs preserve looping.
  const handleSpanSeeking = useCallback(() => {
    const span = spanRef.current;
    const video = videoRef.current;

    if (!video) {
      return;
    }

    if (
      span &&
      video.currentTime >= span.startSeconds - SPAN_SEEK_TOLERANCE_SECONDS &&
      video.currentTime <= span.endSeconds
    ) {
      return;
    }

    spanRef.current = null;
    parkedSpanRef.current = null;
    stopSpanWatch();
    retireSpanToken();
  }, [retireSpanToken, stopSpanWatch]);
  const handlePlay = useCallback(() => {
    startSpanWatch();
    publishSpanState();
  }, [publishSpanState, startSpanWatch]);
  const handlePause = useCallback(() => {
    stopSpanWatch();
    publishSpanState();
  }, [publishSpanState, stopSpanWatch]);
  // Handle emptied because load() pauses without a pause event; otherwise controls offer a stop that cannot change
  // anything.
  const handleEmptied = useCallback(() => {
    publishCopyAvailability();
    publishSpanState();
  }, [publishCopyAvailability, publishSpanState]);
  // Handle ended explicitly for spans ending at clip duration because pause has already stopped the frame watcher.
  const handleSpanEnded = useCallback(() => {
    const span = spanRef.current;
    const video = videoRef.current;

    if (!span || !video) {
      return;
    }

    seekWithinSpan(video, span.startSeconds);
    void video.play().catch(() => {});
    // Publish resumed playback in the same task as end wrapping so the control icon cannot blink between pause and
    // play events.
    publishSpanState();
  }, [publishSpanState, seekWithinSpan]);
  // Pause on unmount or hidden Activity cleanup so audio cannot outlive reachable controls. Subscribe by fixed
  // item key and consume pending requests on show.
  useMountEffect(() => {
    const video = videoRef.current;
    const itemKey = source.itemKey;
    const consumeSpanRequest = (): void => {
      const request = getVideoSpanPlaybackRequest();

      if (!request || request.itemKey !== itemKey) {
        return;
      }

      // Retired whether or not it is still honourable, so a request the user has moved on
      // from cannot wait around for the next player that shows this clip.
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
      stopSpanWatch();
      video?.pause();
      // Report nothing stoppable while hidden, but retain loop identity for native playback after showing.
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

    if (!video || pendingRefreshRef.current) {
      return;
    }

    if (automaticRefreshUsedRef.current) {
      setHasFailed(true);
      return;
    }

    automaticRefreshUsedRef.current = true;
    const accountEpoch = getAuthSession().accountEpoch;
    const refresh = refreshProtectedMediaCookie();
    pendingRefreshRef.current = refresh;

    const finishRefresh = (refreshed: boolean): void => {
      if (pendingRefreshRef.current !== refresh) {
        return;
      }

      pendingRefreshRef.current = null;

      if (
        videoRef.current !== video ||
        getAuthSession().accountEpoch !== accountEpoch ||
        (isItemCurrent && !isItemCurrent(source.itemKey))
      ) {
        return;
      }

      if (refreshed) {
        video.load();
      } else {
        setHasFailed(true);
      }
    };

    void refresh.then(finishRefresh, () => finishRefresh(false));
  }, [isItemCurrent, publishCopyAvailability, source.itemKey]);
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

    // After load() recovery, restore the armed span's position without starting audio or running a nudge that
    // would retire it.
    if (spanRef.current) {
      seekWithinSpan(video, spanRef.current.startSeconds);
      return;
    }

    if (!video.paused || video.currentTime > 0) {
      return;
    }

    try {
      video.currentTime = FIRST_FRAME_SEEK_SECONDS;
    } catch {
      // Some browsers throw if the seekable range is not populated yet; the poster stays up.
    }
  }, [playSpan, seekWithinSpan]);
  const handleRetry = useCallback(() => {
    const video = videoRef.current;

    if (!video) {
      return;
    }

    automaticRefreshUsedRef.current = false;
    pendingRefreshRef.current = null;
    setHasFailed(false);
    video.load();
    publishCopyAvailability();
  }, [publishCopyAvailability]);
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (onContextMenu) {
        event.preventDefault();
        onContextMenu(event.clientX, event.clientY);
      }
    },
    [onContextMenu]
  );

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
        {dragItem ? (
          // Keep drag grips unfocusable so KeyboardSensor cannot start invisible Enter/Space drags.
          <Badge
            ref={setDragHandleRef}
            {...listeners}
            cursor={isDragging ? 'grabbing' : 'grab'}
            // Top-center: every corner belongs to some browser's own video
            // overlays (fullscreen, picture-in-picture), and the bottom strip
            // is the native control bar.
            insetInlineStart="50%"
            position="absolute"
            size="xs"
            title={t('widgets.preview.dragVideo')}
            top="2"
            touchAction="none"
            transform="translateX(-50%)"
            variant="solid"
            // Above the failure overlay: a clip that cannot play can still be
            // dragged into an input, which only needs its name.
            zIndex="2"
          >
            <GripHorizontalIcon aria-hidden="true" size={12} />
          </Badge>
        ) : null}
        {/* User-provided gallery videos do not include a caption-track contract. */}
        {/* oxlint-disable-next-line jsx-a11y/media-has-caption */}
        <video
          ref={setVideoRef}
          aria-label={source.label}
          controls
          playsInline
          poster={source.poster}
          preload="metadata"
          src={source.src}
          style={VIDEO_STYLE}
          onCanPlay={publishCopyAvailability}
          onEmptied={handleEmptied}
          onError={handleVideoError}
          onLoadedData={publishCopyAvailability}
          onEnded={handleSpanEnded}
          onLoadedMetadata={handleLoadedMetadata}
          onPause={handlePause}
          onPlay={handlePlay}
          onPlaying={publishCopyAvailability}
          onResize={publishCopyAvailability}
          onSeeked={publishCopyAvailability}
          onSeeking={handleSpanSeeking}
          onTimeUpdate={enforceSpan}
          onWaiting={publishCopyAvailability}
        />
        {hasFailed ? (
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
              zIndex="1"
            >
              <Text color="white" fontSize="sm" fontWeight="semibold">
                {t('widgets.preview.videoFailed')}
              </Text>
              <Button aria-label={t('widgets.preview.videoRetry')} size="sm" onClick={handleRetry}>
                {t('widgets.preview.videoRetry')}
              </Button>
            </Flex>
          </>
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

export type PreviewVideoFrameCopyFailureReason =
  | 'clipboard-failed'
  | 'draw-failed'
  | 'encode-failed'
  | 'not-ready'
  | 'stale'
  | 'unsupported';

export type PreviewVideoFrameCopyResult = { ok: true } | { ok: false; reason: PreviewVideoFrameCopyFailureReason };

export interface PreviewVideoFrameController {
  copyCurrentFrame(): Promise<PreviewVideoFrameCopyResult>;
  isCopyAvailable(): boolean;
  readonly itemKey: GalleryItemKey;
}

const isVideoFrameReady = (video: HTMLVideoElement | null): video is HTMLVideoElement =>
  Boolean(
    video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0 && video.videoHeight > 0
  );

const isVideoFrameCopyAvailable = (video: HTMLVideoElement | null): boolean =>
  typeof globalThis.ClipboardItem !== 'undefined' &&
  typeof navigator.clipboard?.write === 'function' &&
  isVideoFrameReady(video);

const encodeCanvasPng = (canvas: HTMLCanvasElement): Promise<Blob | null> =>
  new Promise((resolve, reject) => {
    try {
      canvas.toBlob(resolve, 'image/png');
    } catch (error: unknown) {
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });

const isCanvasSecurityError = (error: unknown): boolean =>
  error instanceof DOMException && error.name === 'SecurityError';

/** A trimmed window of a clip, in seconds, as the Video panel's play buttons request it. */
interface VideoSpan {
  endSeconds: number;
  startSeconds: number;
}

/**
 * Allow slight seek undershoot because decoders may not land exactly at span start; own wraps must not retire the
 * loop.
 */
const SPAN_SEEK_TOLERANCE_SECONDS = 0.05;

/**
 * Height of the native video controls, measured up from the frame's bottom edge; touches there never swipe. Tall
 * enough for Chrome Android's timeline, which sits above its button row.
 */
const VIDEO_CONTROL_BAND_PX = 72;

/** Fills the fitted frame, which already has the image's aspect ratio. */
const PLACEHOLDER_IMAGE_STYLE: CSSProperties = {
  height: '100%',
  inset: 0,
  objectFit: 'contain',
  pointerEvents: 'none',
  position: 'absolute',
  width: '100%',
};

/** How long a span waits for an element that could not act on it yet. */
const PARKED_SPAN_TTL_MS = 15_000;

// Seek far enough from zero to clear poster state while staying within frame zero.
const FIRST_FRAME_SEEK_SECONDS = 0.0001;

const VIDEO_STYLE: CSSProperties = {
  display: 'block',
  height: '100%',
  objectFit: 'contain',
  width: '100%',
};

const VIDEO_FAILURE_POSTER_STYLE: CSSProperties = {
  height: '100%',
  inset: 0,
  objectFit: 'contain',
  position: 'absolute',
  width: '100%',
};

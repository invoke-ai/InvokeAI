/* eslint-disable react/refs, react/immutability */
import type { GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';
import type * as VideoPlayerModule from '@platform/ui/video-player/VideoPlayer';

import { Badge, Flex, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { getAuthSession, refreshProtectedMediaCookie } from '@features/identity';
import { canCopyImage, copyImage } from '@platform/browser/clipboard';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { GripHorizontalIcon } from 'lucide-react';
import {
  lazy,
  Suspense,
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

// Most selections are images, and other surfaces share the player: it loads with the first video shown, not with the
// Preview, so it costs the Preview's own load no request. Its bar exists only once it has loaded, so the stage reads
// the bar test from the loaded module.
let loadedVideoPlayer: typeof VideoPlayerModule | null = null;
const VideoPlayer = lazy(() =>
  import('@platform/ui/video-player/VideoPlayer').then((module) => {
    loadedVideoPlayer = module;

    return { default: module.VideoPlayer };
  })
);

export type PreviewMediaSource =
  | { itemKey: GalleryItemKey; kind: 'image'; source: StreamingImageSource }
  | { fps?: number; itemKey: GalleryItemKey; kind: 'video'; label: string; poster: string; src: string };

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
  /** Whether videos loop, when the caller remembers it; otherwise each player starts looping. */
  videoLoop?: boolean;
  onVideoLoopChange?: (loop: boolean) => void;
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
        loop={props.videoLoop}
        onContextMenu={props.onContextMenu}
        onCopyAvailabilityChange={props.onVideoCopyAvailabilityChange}
        onLoopChange={props.onVideoLoopChange}
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
}: Omit<
  PreviewFrameProps,
  | 'isItemCurrent'
  | 'onVideoCopyAvailabilityChange'
  | 'onVideoLoopChange'
  | 'source'
  | 'videoControllerRef'
  | 'videoLoop'
> & {
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

/**
 * The player remounts per item, so its audio settings live here for the page session: the first clip starts at half
 * volume (the browser default of full volume is usually too loud) and later clips keep the viewer's last choice.
 * A viewer preference, not account data, so it is deliberately not cleared on account changes.
 */
const previewVideoAudio = { muted: false, volume: 0.5 };

const rememberPreviewVideoAudio = (event: SyntheticEvent<HTMLVideoElement>): void => {
  previewVideoAudio.muted = event.currentTarget.muted;
  previewVideoAudio.volume = event.currentTarget.volume;
};

const PreviewVideo = ({
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
}: {
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
}) => {
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
  // The control bar owns its touches: seeking and volume must never swipe. So do the player's popups, whose events
  // reach the stage through their portals. A fullscreen player is out of the carousel entirely; navigating would
  // unmount it and drop fullscreen.
  const handleStagePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const target = event.target;

      if (
        !(target instanceof Element) ||
        !event.currentTarget.contains(target) ||
        loadedVideoPlayer?.isVideoPlayerControlsTarget(target) ||
        target.ownerDocument.fullscreenElement?.contains(target)
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
      if (video) {
        video.muted = previewVideoAudio.muted;
        video.volume = previewVideoAudio.volume;
      }
      onCopyAvailabilityChange?.(source.itemKey, isVideoFrameCopyAvailable(video));
    },
    [onCopyAvailabilityChange, source.itemKey]
  );
  const copyCurrentFrame = useCallback(async (): Promise<PreviewVideoFrameCopyResult> => {
    const video = videoRef.current;

    if (!canCopyImage()) {
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

    try {
      await copyImage(blob);
    } catch {
      return { ok: false, reason: 'clipboard-failed' };
    }

    return isCurrent() ? { ok: true } : { ok: false, reason: 'stale' };
  }, [isItemCurrent, source.itemKey]);
  const isCopyAvailable = useCallback(() => isVideoFrameCopyAvailable(videoRef.current), []);
  // The span a request armed, which the player plays as its playback window. Handlers read the ref: a seek's event
  // can arrive before the render that arms its span.
  const [span, setSpan] = useState<VideoSpan | null>(null);
  const spanRef = useRef<VideoSpan | null>(null);
  // Expire spans parked before media readiness so retries or hidden remounts cannot start stale audio later.
  const parkedSpanRef = useRef<{ expiresAt: number; span: VideoSpan; token: number } | null>(null);
  // Keep request identity through pause/hide while the span stays armed; retire it when the playhead leaves or a
  // newer request arrives.
  const spanTokenRef = useRef<number | null>(null);
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

      // Defer seeks until metadata and element attachment exist; resume from loadedmetadata or remount.
      if (!video || video.readyState < HTMLMediaElement.HAVE_METADATA) {
        // The deadline belongs to the gesture, not to this attempt: minting a fresh one on
        // every re-park would let hide/show cycles on a clip that never loads extend it
        // indefinitely, which is the one thing the deadline exists to stop.
        parkedSpanRef.current = { expiresAt: expiresAt ?? Date.now() + PARKED_SPAN_TTL_MS, span: request, token };
        return;
      }

      parkedSpanRef.current = null;
      const duration = Number.isFinite(video.duration) ? video.duration : null;
      const endSeconds = duration === null ? request.endSeconds : Math.min(request.endSeconds, duration);
      const startSeconds = Math.max(0, Math.min(request.startSeconds, endSeconds));
      // Arm any nonempty clamped span, including two-frame trims. Collapsed or inverted bounds are no window: the clip
      // plays from their start as the loop toggle says.
      const next = endSeconds > startSeconds ? { endSeconds, startSeconds } : null;

      // New requests replace prior span ownership; report only an armed span.
      retireSpanToken();
      armSpan(next);

      if (next !== null) {
        spanTokenRef.current = token;
      }

      seekVideo(video, startSeconds);
      // Autoplay policy can refuse an unmuted play(). That leaves the clip parked on the first selected frame with
      // the span still armed, so pressing play there plays the selection rather than the whole clip.
      void video.play().catch(() => {});
      publishSpanState();
    },
    [armSpan, publishSpanState, retireSpanToken]
  );
  // Retire spans only when seeks land outside them; own-seek markers fail for no-op or delayed seeks. In-span scrubs,
  // and the player's own wraps, rests and replays, keep the span. The player never acts on a seek, so the retirement
  // needs no particular timing.
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
  const playerFallback = useMemo(
    () => (
      <img
        aria-hidden="true"
        alt=""
        draggable={false}
        height={frameHeight}
        src={source.poster}
        style={VIDEO_PLAYER_FALLBACK_STYLE}
        width={frameWidth}
      />
    ),
    [frameHeight, frameWidth, source.poster]
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
        <Suspense fallback={playerFallback}>
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
        </Suspense>
        {dragItem ? (
          // Keep drag grips unfocusable so KeyboardSensor cannot start invisible Enter/Space drags.
          <Badge
            ref={setDragHandleRef}
            {...listeners}
            cursor={isDragging ? 'grabbing' : 'grab'}
            // Top-center, clear of the control bar along the bottom edge. Outside the player, so fullscreen leaves it
            // behind, where a drag could reach no drop target; above the failure overlay, since a clip that cannot
            // play can still be dragged into an input, which only needs its name.
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
  canCopyImage() && isVideoFrameReady(video);

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

const seekVideo = (video: HTMLVideoElement, seconds: number): void => {
  try {
    video.currentTime = seconds;
  } catch {
    // Some browsers throw while the seekable range is still empty; the element keeps its position.
  }
};

/** A trimmed window of a clip, in seconds, as the Video panel's play buttons request it. */
interface VideoSpan {
  endSeconds: number;
  startSeconds: number;
}

/**
 * Allow slight seek undershoot because decoders may not land exactly at span start; the player's own wraps must not
 * retire the span.
 */
const SPAN_SEEK_TOLERANCE_SECONDS = 0.05;

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

/** The poster stands in, as the player would show it, while the player loads. */
const VIDEO_PLAYER_FALLBACK_STYLE: CSSProperties = {
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

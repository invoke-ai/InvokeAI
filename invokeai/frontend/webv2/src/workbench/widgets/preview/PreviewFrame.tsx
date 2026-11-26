/* eslint-disable react/refs, react/immutability */
import type { GalleryItemKey, GalleryItemRef } from '@features/gallery';
import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';
import type { VideoFrameCopyFailureReason, VideoFrameCopyResult } from '@platform/ui/video-player/videoFrameUtils';

import { Badge } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { GripHorizontalIcon } from 'lucide-react';
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useId,
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
import { usePreviewLoupe, type PreviewLoupeControls, type PreviewZoomState } from './usePreviewLoupe';
import { usePreviewSwipe, type PreviewSwipeNavigation } from './usePreviewSwipe';

// Most selections are images, and other surfaces share the player: keep the video shell (drag grip, context menu,
// swipe) eager and lazy-load only the inner player (VideoPlayer + videoFrameUtils) on the first video shown.
const LazyPreviewVideoPlayer = lazy(() => import('./PreviewVideoFrame'));

export type PreviewMediaSource =
  | { itemKey: GalleryItemKey; kind: 'image'; source: StreamingImageSource }
  | { fps?: number; itemKey: GalleryItemKey; kind: 'video'; label: string; poster: string; src: string };

export type PreviewVideoFrameCopyFailureReason = VideoFrameCopyFailureReason;

export type PreviewVideoFrameCopyResult = VideoFrameCopyResult;

export interface PreviewVideoFrameController {
  copyCurrentFrame(): Promise<PreviewVideoFrameCopyResult>;
  isCopyAvailable(): boolean;
  readonly itemKey: GalleryItemKey;
}

interface PreviewFrameProps {
  children?: ReactNode;
  dragItem?: GalleryItemRef;
  frameHeight: number;
  frameWidth: number;
  /** Keep the last denoise frame until source decoding completes, then notify onSourceLoaded. */
  holdSource?: StreamingImageSource | null;
  /** A settled image failed to load. */
  onSourceError?: () => void;
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
        padding={props.padding}
        paddingBottom={props.paddingBottom}
        source={props.source}
        swipe={props.swipe}
        videoControllerRef={props.videoControllerRef}
        onContextMenu={props.onContextMenu}
        onCopyAvailabilityChange={props.onVideoCopyAvailabilityChange}
        onLoopChange={props.onVideoLoopChange}
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
  onSourceError,
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
  const isDragDisabled = !dragItem || isLive || loupe.isPannable;
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
    isZoomed: loupe.isPannable,
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
  // Reset zoom in place when the displayed image changes (or goes live) — a remount would flash the frame on every
  // selection.
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
  // `image-rendering` stays unset while the loupe is enabled so the loupe can toggle pixelated rendering on the content
  // box (it inherits to the img).
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

      // Only tracked while a hold is up: a live frame is a new data URL every step, and settling each one would
      // re-render the frame per step for nothing.
      if (src === null || !holdSource) {
        return;
      }

      setSettledSrc(src);
      onSourceLoaded?.(src);
    },
    [holdSource, onSourceLoaded, placeholderSrc]
  );
  const handleSourceError = useCallback(
    (event: SyntheticEvent<HTMLImageElement>) => {
      handleSourceSettled(event);
      onSourceError?.();
    },
    [handleSourceSettled, onSourceError]
  );
  // A hold arriving after the image already decoded (a cached image, or the element reused across a source swap) would
  // never see a load event.
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
        onError={handleSourceError}
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
      cursor={loupe.isPannable ? 'grab' : undefined}
      fill="flex"
      padding={padding}
      paddingBottom={paddingBottom}
      // Suppress browser gestures across the loupe's entire stage; live renders without a loupe retain native behavior.
      touchAction={isLive ? undefined : 'none'}
      {...loupe.stageProps}
      onPointerDown={handleStagePointerDown}
    >
      {/* No comparison drops over live renders: comparison would pause follow and expose a stale selection. */}
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
  const isControlsTargetRef = useRef<((target: Element) => boolean) | null>(null);
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
        isControlsTargetRef.current?.(target) ||
        target.ownerDocument.fullscreenElement?.contains(target)
      ) {
        return;
      }

      handleSwipePointerDown(event);
    },
    [handleSwipePointerDown]
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
          <LazyPreviewVideoPlayer
            frameHeight={frameHeight}
            frameWidth={frameWidth}
            isControlsTargetRef={isControlsTargetRef}
            isItemCurrent={isItemCurrent}
            loop={loopPreference}
            source={source}
            videoControllerRef={videoControllerRef}
            onCopyAvailabilityChange={onCopyAvailabilityChange}
            onLoopChange={onLoopChange}
          />
        </Suspense>
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

/** Fills the fitted frame, which already has the image's aspect ratio. */
const PLACEHOLDER_IMAGE_STYLE: CSSProperties = {
  height: '100%',
  inset: 0,
  objectFit: 'contain',
  pointerEvents: 'none',
  position: 'absolute',
  width: '100%',
};

/** The poster stands in, as the player would show it, while the player loads. */
const VIDEO_PLAYER_FALLBACK_STYLE: CSSProperties = {
  display: 'block',
  height: '100%',
  objectFit: 'contain',
  width: '100%',
};

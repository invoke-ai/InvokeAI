import type { GalleryItemKey, GalleryItemRef } from '@features/gallery';
/* eslint-disable react/react-compiler */
import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';

import { Badge, Flex, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { getAuthSession, refreshProtectedMediaCookie } from '@features/identity';
import { Button } from '@platform/ui/Button';
import {
  useCallback,
  useId,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent,
  type ReactNode,
  type Ref,
} from 'react';
import { useTranslation } from 'react-i18next';

import { PreviewCompareDropZone } from './PreviewCompareDropZone';
import { FittedFrame, PreviewStage } from './PreviewStage';
import { usePreviewLoupe, type PreviewLoupeControls } from './usePreviewLoupe';

export type PreviewMediaSource =
  | { itemKey: GalleryItemKey; kind: 'image'; source: StreamingImageSource }
  | { itemKey: GalleryItemKey; kind: 'video'; label: string; poster: string; src: string };

interface PreviewFrameProps {
  children?: ReactNode;
  dragItem?: GalleryItemRef;
  frameHeight: number;
  frameWidth: number;
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  isLive: boolean;
  /**
   * Static caption over a live frame — used only by the multi-session tiles to
   * name the GPU. The single live preview carries no badge at all: the frame
   * is styled exactly like a finished item, and progress readouts belong to
   * the footer and the top bar rail.
   */
  liveBadgeLabel?: string;
  loupeControlsRef?: Ref<PreviewLoupeControls>;
  onContextMenu?: (x: number, y: number) => void;
  onVideoCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  padding?: string;
  paddingBottom?: string;
  shouldAntialiasLiveImage: boolean;
  source: PreviewMediaSource | null;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
  variant: 'framed' | 'inset';
}

export const PreviewFrame = (props: PreviewFrameProps) => {
  if (props.source?.kind === 'video') {
    return (
      <PreviewVideo
        key={props.source.itemKey}
        frameHeight={props.frameHeight}
        frameWidth={props.frameWidth}
        isItemCurrent={props.isItemCurrent}
        onContextMenu={props.onContextMenu}
        onCopyAvailabilityChange={props.onVideoCopyAvailabilityChange}
        padding={props.padding}
        paddingBottom={props.paddingBottom}
        source={props.source}
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
  isLive,
  liveBadgeLabel,
  loupeControlsRef,
  onContextMenu,
  padding,
  paddingBottom,
  shouldAntialiasLiveImage,
  source,
  variant,
}: Omit<PreviewFrameProps, 'isItemCurrent' | 'onVideoCopyAvailabilityChange' | 'source' | 'videoControllerRef'> & {
  source: StreamingImageSource | null;
}) => {
  const { t } = useTranslation();
  const loupe = usePreviewLoupe({
    controlsRef: loupeControlsRef,
    enabled: variant === 'framed' && !isLive,
    naturalWidth: frameWidth,
  });
  const dragData = useMemo(() => (dragItem ? getGalleryItemDragData([dragItem]) : undefined), [dragItem]);
  const isDragDisabled = !dragItem || isLive || loupe.isZoomed;
  const disabledDragId = useId();
  const {
    isDragging,
    listeners,
    setNodeRef: setDragNodeRef,
  } = useDraggable({
    data: dragData,
    disabled: isDragDisabled,
    id: dragItem ? getGalleryItemDragId(dragItem, 'preview-frame') : `preview-frame:disabled:${disabledDragId}`,
  });
  const setContentRef = useCallback(
    (element: HTMLDivElement | null) => {
      setDragNodeRef(element);

      if (loupe.contentRef) {
        loupe.contentRef.current = element;
      }
    },
    [loupe.contentRef, setDragNodeRef]
  );

  // Reset zoom in place when the displayed image changes (or goes live) — a
  // remount would flash the frame on every selection.
  loupe.syncDisplayedSource(variant === 'framed' && !isLive && source ? source.src : null);
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
      width: '100%',
    }),
    [isLive, shouldAntialiasLiveImage]
  );
  const media = source ? (
    <img
      alt={source.alt}
      draggable={false}
      height={frameHeight}
      src={source.src}
      style={imageStyle}
      width={frameWidth}
    />
  ) : null;
  const liveBadge =
    isLive && liveBadgeLabel !== undefined ? (
      <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
        {liveBadgeLabel}
      </Badge>
    ) : null;

  if (variant === 'inset') {
    return (
      // Live tiles all reserve the chrome inset, so in a multi-session grid the
      // bottom row reserves room it does not need. Its own dot grid still fills
      // the cell, so only the fitted frame sits a little lower.
      <PreviewStage fill="parent">
        {source ? (
          <FittedFrame frameHeight={frameHeight} frameWidth={frameWidth}>
            {media}
            {liveBadge}
          </FittedFrame>
        ) : (
          children
        )}
      </PreviewStage>
    );
  }

  return (
    <PreviewStage
      ref={loupe.stageRefCallback}
      cursor={loupe.isZoomed ? 'grab' : undefined}
      fill="flex"
      padding={padding}
      paddingBottom={paddingBottom}
      {...loupe.stageProps}
    >
      {/* Never armed over a live render: arming a comparison pauses live-follow
          and would swap the in-progress image for a compare of the stale hidden
          selection. The inset live frame never offered this either. */}
      {isLive ? null : <PreviewCompareDropZone currentImageName={dragItem?.kind === 'image' ? dragItem.name : null} />}
      <FittedFrame
        ref={setContentRef}
        {...listeners}
        bg="transparent"
        cursor={isDragDisabled ? undefined : isDragging ? 'grabbing' : 'grab'}
        frameHeight={frameHeight}
        frameWidth={frameWidth}
        opacity={isDragging ? 0.55 : undefined}
        touchAction={isDragDisabled ? undefined : 'none'}
        onContextMenu={onContextMenu ? handleContextMenu : undefined}
      >
        {media}
        {liveBadge}
      </FittedFrame>
      {loupe.zoomPercent !== null ? (
        <Badge
          aria-label={t('widgets.preview.resetZoom')}
          as="button"
          bottom="2"
          cursor="pointer"
          position="absolute"
          right="2"
          size="xs"
          title={t('widgets.preview.resetZoom')}
          variant="solid"
          zIndex="1"
          onClick={loupe.reset}
        >
          {loupe.zoomPercent}%
        </Badge>
      ) : null}
    </PreviewStage>
  );
};

const PreviewVideo = ({
  frameHeight,
  frameWidth,
  isItemCurrent,
  onContextMenu,
  onCopyAvailabilityChange,
  padding,
  paddingBottom,
  source,
  videoControllerRef,
}: {
  frameHeight: number;
  frameWidth: number;
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  onContextMenu?: (x: number, y: number) => void;
  onCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  padding?: string;
  paddingBottom?: string;
  source: Extract<PreviewMediaSource, { kind: 'video' }>;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
}) => {
  const { t } = useTranslation();
  const videoRef = useRef<HTMLVideoElement | null>(null);
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
    <PreviewStage fill="flex" padding={padding} paddingBottom={paddingBottom}>
      <FittedFrame
        bg="black"
        frameHeight={frameHeight}
        frameWidth={frameWidth}
        onContextMenu={onContextMenu ? handleContextMenu : undefined}
      >
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
          onEmptied={publishCopyAvailability}
          onError={handleVideoError}
          onLoadedData={publishCopyAvailability}
          onPlaying={publishCopyAvailability}
          onResize={publishCopyAvailability}
          onSeeked={publishCopyAvailability}
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

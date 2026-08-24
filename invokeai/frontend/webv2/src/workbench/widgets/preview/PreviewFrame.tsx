import type { GalleryItemKey, GalleryItemRef } from '@features/gallery';
/* eslint-disable react/react-compiler */
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
  /**
   * No live frame carries a caption of any kind. The frame is styled exactly
   * like a finished item so nothing about it moves when denoising ends, and
   * every progress readout — including which device is rendering — belongs to
   * the footer island or the top bar rail.
   */
  isLive: boolean;
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
        dragItem={props.dragItem}
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
  if (variant === 'inset') {
    return (
      // Live tiles all reserve the chrome inset, so in a multi-session grid the
      // bottom row reserves room it does not need. Its own dot grid still fills
      // the cell, so only the fitted frame sits a little lower.
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
  dragItem,
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
  dragItem?: GalleryItemRef;
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
  // The `<video controls>` surface cannot be the drag handle the way the image
  // frame is: the pointer-move activation threshold would turn every native
  // seek-bar scrub into a drag. A corner grip carries the same gallery-item
  // payload instead, so the preview and the gallery thumbnail drop the same
  // thing everywhere.
  const dragData = useMemo(() => (dragItem ? getGalleryItemDragData([dragItem]) : undefined), [dragItem]);
  const disabledDragId = useId();
  const {
    isDragging,
    listeners,
    setNodeRef: setDragHandleRef,
  } = useDraggable({
    data: dragData,
    disabled: !dragItem,
    id: dragItem ? getGalleryItemDragId(dragItem, 'preview-frame') : `preview-frame:disabled:${disabledDragId}`,
  });
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
  // Playback must not outlive this view being on screen. A widget the shell
  // keeps mounted behind a layout switch is hidden with `display: none`, which
  // does not stop media — the clip would keep running, with audio, behind a
  // layout the user moved away from and with no reachable controls. Effects are
  // torn down whenever the subtree stops being shown, whether that is a real
  // unmount or a hidden keep-alive boundary, so pausing here covers both without
  // this component needing to know which one happened.
  useMountEffect(() => {
    const video = videoRef.current;

    return () => {
      video?.pause();
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
  // The preview keeps `poster` — the 256px WebP gallery thumbnail — as its instant
  // placeholder, but the browser goes on showing it well after the video's own first frame is
  // decoded and ready: by `loadedmetadata` the element is already at readyState 4 with
  // `resize` fired, and the poster survives only because the HTML spec's *show-poster flag* is
  // still set. Upscaled across the whole stage, that 256px still is what reads as fuzzy.
  // Seeking clears the show-poster flag, and that — not any new decoding — is what puts the
  // native-resolution frame on screen. Measured on a 1280x720 clip, Laplacian variance of the
  // painted stage goes from 254 to 576.
  //
  // Dropping the `poster` attribute instead does not work: the flag stays set with nothing
  // left to draw, and the stage paints black until the user presses play.
  //
  // The nudge is not free. It pulls more of the clip than `preload="metadata"` alone would —
  // measured on a 7MB/20s clip, one 1MB range request becomes three (~3.1MB), which then
  // settles rather than running away. A full-resolution poster served by the backend would buy
  // the same frame without the extra range traffic, if that ever becomes worth the endpoint.
  //
  // The guard skips the nudge when the user hit play before metadata arrived — measured, that
  // is the one interleaving where `loadedmetadata` observes `paused === false`. It deliberately
  // does not claim to protect a mid-playback viewer from a reload: `load()` is the only thing
  // that refires `loadedmetadata` on this element, and it has already reset the position to 0
  // and `paused` to true by the time the handler runs. The `currentTime` clause is
  // belt-and-braces for an engine that does not reset it.
  const handleFirstFrameSeek = useCallback(() => {
    const video = videoRef.current;

    if (!video || !video.paused || video.currentTime > 0) {
      return;
    }

    try {
      video.currentTime = FIRST_FRAME_SEEK_SECONDS;
    } catch {
      // Some browsers throw if the seekable range is not populated yet; the poster stays up.
    }
  }, []);
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
        opacity={isDragging ? 0.55 : undefined}
        onContextMenu={onContextMenu ? handleContextMenu : undefined}
      >
        {dragItem ? (
          // Not a button: like the gallery thumbnail and the image frame, the
          // drag surface must stay unfocusable — a focusable activator lets the
          // KeyboardSensor start an invisible drag on Enter/Space that Tab then
          // DROPS on the closest-center droppable.
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
          onEmptied={publishCopyAvailability}
          onError={handleVideoError}
          onLoadedData={publishCopyAvailability}
          onLoadedMetadata={handleFirstFrameSeek}
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

// Far enough from zero for browsers to treat it as a real seek, small enough that playback
// still starts on frame 0 at any sane frame rate.
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

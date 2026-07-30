import type { GalleryItemKey, GalleryItemRef } from '@features/gallery';
/* eslint-disable react/react-compiler */
import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';

import { Badge, Box, Button, Flex, Text, type SystemStyleObject } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { getAuthSession, refreshProtectedMediaCookie } from '@features/identity';
import {
  useCallback,
  useId,
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
import { PreviewLiveOverlay } from './PreviewLiveReadout';
import { usePreviewLoupe, type PreviewLoupeControls } from './usePreviewLoupe';

/**
 * The preview's image surface: a dot-grid backdrop with an aspect-fitted,
 * shadowed frame. Owns everything drawn over the image (live badge, and later
 * the loupe, progress hairline, and drop-to-compare affordance) so the widget
 * shell never grows frame-specific rendering.
 */

export const previewGridCss = {
  backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1.5px)',
  backgroundPosition: 'center',
  backgroundRepeat: 'repeat',
  backgroundSize: '24px 24px',
} as const;

export const getFittedFrameCss = (width: number, height: number): SystemStyleObject => ({
  aspectRatio: `${width} / ${height}`,
  height: 'auto',
  maxHeight: '100%',
  maxWidth: '100%',
  width: `min(100cqw, calc(100cqh * ${width / height}))`,
});

export type PreviewMediaSource =
  | { itemKey: GalleryItemKey; kind: 'image'; source: StreamingImageSource }
  | { itemKey: GalleryItemKey; kind: 'video'; label: string; poster: string; src: string };

interface PreviewFrameProps {
  /** Rendered instead of the fitted frame when there is no image source (inset variant only). */
  children?: ReactNode;
  /** Saved gallery image represented by this frame. Live progress frames are never draggable. */
  dragItem?: GalleryItemRef;
  frameHeight: number;
  frameWidth: number;
  /** Live selected-key read used by protected-video retry guards. */
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  isLive: boolean;
  liveBadgeLabel: string;
  /** When set, the static live badge is replaced by the live progress readout for this run. */
  liveQueueItemId?: string | null;
  /** Imperative zoom controls, for hotkeys registered by the widget shell. */
  loupeControlsRef?: Ref<PreviewLoupeControls>;
  onContextMenu?: (x: number, y: number) => void;
  padding?: string;
  shouldAntialiasLiveImage: boolean;
  source: PreviewMediaSource | null;
  /** `framed` = bordered surface for a selected item; `inset` = flush surface for the empty state. */
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
        padding={props.padding}
        source={props.source}
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
  liveQueueItemId,
  loupeControlsRef,
  onContextMenu,
  padding,
  shouldAntialiasLiveImage,
  source,
  variant,
}: Omit<PreviewFrameProps, 'isItemCurrent' | 'source'> & { source: StreamingImageSource | null }) => {
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
  const liveBadge = isLive ? (
    typeof liveQueueItemId === 'string' ? (
      <PreviewLiveOverlay queueItemId={liveQueueItemId} />
    ) : (
      <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
        {liveBadgeLabel}
      </Badge>
    )
  ) : null;

  if (variant === 'inset') {
    return (
      <Flex
        align="center"
        backgroundColor="bg.inset"
        color="fg.grid"
        containerType="size"
        css={previewGridCss}
        h="full"
        justify="center"
        w="full"
      >
        {source ? (
          <Box
            borderColor={isLive ? 'accent.solid' : 'border.emphasized'}
            borderWidth="1px"
            boxShadow="0 24px 80px rgba(0,0,0,0.42)"
            css={getFittedFrameCss(frameWidth, frameHeight)}
            overflow="hidden"
            position="relative"
            rounded="lg"
          >
            <img
              alt={source.alt}
              draggable={false}
              height={frameHeight}
              src={source.src}
              style={imageStyle}
              width={frameWidth}
            />
            {liveBadge}
          </Box>
        ) : (
          children
        )}
      </Flex>
    );
  }

  return (
    <Flex
      ref={loupe.stageRefCallback}
      align="center"
      borderWidth="1px"
      borderColor="border.subtle"
      color="fg.grid"
      containerType="size"
      css={previewGridCss}
      cursor={loupe.isZoomed ? 'grab' : undefined}
      flex="1"
      justify="center"
      minH="0"
      overflow="hidden"
      p={padding}
      position="relative"
      rounded="lg"
      w="full"
      {...loupe.stageProps}
    >
      <PreviewCompareDropZone />
      <Box
        ref={setContentRef}
        {...listeners}
        bg="transparent"
        borderWidth="1px"
        borderColor={isLive ? 'accent.solid' : 'border.emphasized'}
        boxShadow="0 24px 80px rgba(0,0,0,0.42)"
        css={getFittedFrameCss(frameWidth, frameHeight)}
        cursor={isDragDisabled ? undefined : isDragging ? 'grabbing' : 'grab'}
        opacity={isDragging ? 0.55 : undefined}
        overflow="hidden"
        position="relative"
        rounded="lg"
        touchAction={isDragDisabled ? undefined : 'none'}
        onContextMenu={onContextMenu ? handleContextMenu : undefined}
      >
        {source ? (
          <img
            alt={source.alt}
            draggable={false}
            height={frameHeight}
            src={source.src}
            style={imageStyle}
            width={frameWidth}
          />
        ) : null}
        {liveBadge}
      </Box>
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
    </Flex>
  );
};

const PreviewVideo = ({
  frameHeight,
  frameWidth,
  isItemCurrent,
  onContextMenu,
  padding,
  source,
}: {
  frameHeight: number;
  frameWidth: number;
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  onContextMenu?: (x: number, y: number) => void;
  padding?: string;
  source: Extract<PreviewMediaSource, { kind: 'video' }>;
}) => {
  const { t } = useTranslation();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const automaticRefreshUsedRef = useRef(false);
  const pendingRefreshRef = useRef<Promise<boolean> | null>(null);
  const [hasFailed, setHasFailed] = useState(false);
  const setVideoRef = useCallback((video: HTMLVideoElement | null) => {
    videoRef.current = video;
  }, []);
  const handleVideoError = useCallback(() => {
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
  }, [isItemCurrent, source.itemKey]);
  const handleRetry = useCallback(() => {
    const video = videoRef.current;

    if (!video) {
      return;
    }

    automaticRefreshUsedRef.current = false;
    pendingRefreshRef.current = null;
    setHasFailed(false);
    video.load();
  }, []);
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
    <Flex
      align="center"
      borderWidth="1px"
      borderColor="border.subtle"
      color="fg.grid"
      containerType="size"
      css={previewGridCss}
      flex="1"
      justify="center"
      minH="0"
      overflow="hidden"
      p={padding}
      position="relative"
      rounded="lg"
      w="full"
    >
      <Box
        bg="black"
        borderWidth="1px"
        borderColor="border.emphasized"
        boxShadow="0 24px 80px rgba(0,0,0,0.42)"
        css={getFittedFrameCss(frameWidth, frameHeight)}
        overflow="hidden"
        position="relative"
        rounded="lg"
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
          onError={handleVideoError}
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
      </Box>
    </Flex>
  );
};

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

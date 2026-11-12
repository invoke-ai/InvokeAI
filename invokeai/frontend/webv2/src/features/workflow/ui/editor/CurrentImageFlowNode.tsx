import type { GalleryItem, GalleryItemRef, GeneratedImageContract } from '@features/gallery/contracts';
import type * as VideoPlayerModule from '@platform/ui/video-player/VideoPlayer';
import type { NodeProps } from '@xyflow/react';

import { Badge, Box, Flex, HStack, Icon, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import {
  getSelectedGalleryItemFromValues,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  toGalleryItemRef,
} from '@features/gallery/contracts';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import {
  useActiveProgressTargets,
  useIsProcessorPaused,
  useProgressImage,
  useQueueItemProgress,
} from '@features/queue/react';
import {
  CURRENT_IMAGE_DEFAULT_HEIGHT,
  CURRENT_IMAGE_DEFAULT_WIDTH,
  clampCurrentImageHeight,
  clampCurrentImageWidth,
} from '@features/workflow/core/types';
import {
  getWorkflowNodeHeaderProps,
  getWorkflowNodeShellProps,
  useIsWorkflowImageExport,
} from '@features/workflow/ui/nodeChrome';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowNotifications, useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { canCopyImage, copyImage } from '@platform/browser/clipboard';
import { Button, IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import {
  imageUrlToStreamingSource,
  progressImageToStreamingSource,
} from '@platform/ui/streaming-image/streamingImageSource';
import { Tooltip } from '@platform/ui/Tooltip';
import { CopyIcon, GripHorizontalIcon } from 'lucide-react';
import {
  lazy,
  memo,
  Suspense,
  useCallback,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
} from 'react';
import { useTranslation } from 'react-i18next';

import type { CurrentImageFlowNode as CurrentImageFlowNodeType } from './flowAdapters';

const KEYBOARD_RESIZE_STEP_PX = 32;
const RING_CIRCUMFERENCE = 37.7;
const INDETERMINATE_DASH_OFFSET = 27.1;
const RESIZE_FOCUS_VISIBLE_STYLE = {
  opacity: 1,
  outline: '2px solid',
  outlineColor: 'accent.solid',
  outlineOffset: '-2px',
} as const;
const RESIZE_HOVER_STYLE = { opacity: 1 } as const;

const LazyVideoPlayer = lazy(() =>
  import('@platform/ui/video-player/VideoPlayer').then((mod) => ({ default: mod.VideoPlayer }))
);

// Keep VideoPlayer typed while still splitting the player chunk until a video is selected.
const PlayerComponent: typeof VideoPlayerModule.VideoPlayer =
  LazyVideoPlayer as unknown as typeof VideoPlayerModule.VideoPlayer;

interface ResizeDragSession {
  pointerId: number;
  startHeight: number;
  startWidth: number;
  startX: number;
  startY: number;
  zoom: number;
}

const resolveDisplayedGalleryItem = (values: Record<string, unknown>): GalleryItem | null => {
  const selected = getSelectedGalleryItemFromValues(values);
  if (selected) {
    return selected;
  }

  const recentImages = Array.isArray(values.recentImages) ? (values.recentImages as GeneratedImageContract[]) : [];
  const latest = recentImages[0];

  return latest ? legacyGeneratedImageToGalleryItem(latest) : null;
};

const isVideoFrameReady = (video: HTMLVideoElement | null): video is HTMLVideoElement =>
  Boolean(
    video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0 && video.videoHeight > 0
  );

const encodeCanvasPng = (canvas: HTMLCanvasElement): Promise<Blob | null> =>
  new Promise((resolve, reject) => {
    try {
      canvas.toBlob(resolve, 'image/png');
    } catch (error: unknown) {
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });

const CurrentImageEditorNode = ({ data, selected }: NodeProps<CurrentImageFlowNodeType>) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const notifications = useWorkflowNotifications();
  const galleryValues = useWorkflowProjectSelector((project) => project.galleryValues);
  const isWorkflowRunning = useWorkflowProjectSelector((project) => project.isWorkflowRunning);
  const progressImage = useProgressImage();
  const activeTargets = useActiveProgressTargets();
  const isProcessorPaused = useIsProcessorPaused();

  // Gate render status strictly on live execution (`progressImage`, `activeTargets`, or an unpaused running workflow)
  // and read percentage from the active queue item target so completed backend entries never keep spinning.
  const isGenerating = Boolean(progressImage) || activeTargets.length > 0 || (isWorkflowRunning && !isProcessorPaused);
  const activeQueueItemId = activeTargets[0]?.queueItemId ?? '';
  const activeProgress = useQueueItemProgress(isGenerating ? activeQueueItemId : '');
  const progressPercent =
    isGenerating && typeof activeProgress?.percentage === 'number' && activeProgress.percentage > 0
      ? Math.round(Math.min(100, Math.max(0, activeProgress.percentage * 100)))
      : null;

  const node = data.documentNode;
  const savedWidth = node.data.width ?? CURRENT_IMAGE_DEFAULT_WIDTH;
  const savedHeight = node.data.height ?? CURRENT_IMAGE_DEFAULT_HEIGHT;
  const [liveSize, setLiveSize] = useState<{ height: number; width: number } | null>(null);
  const displayWidth = liveSize?.width ?? savedWidth;
  const displayHeight = liveSize?.height ?? savedHeight;
  const isCustomSize = displayWidth !== CURRENT_IMAGE_DEFAULT_WIDTH || displayHeight !== CURRENT_IMAGE_DEFAULT_HEIGHT;

  const displayedItem = useMemo(() => resolveDisplayedGalleryItem(galleryValues), [galleryValues]);
  const isShowingVideo = !isGenerating && displayedItem?.kind === 'video';
  const dragRef = useMemo<GalleryItemRef | null>(
    () => (displayedItem && !isGenerating ? toGalleryItemRef(displayedItem) : null),
    [displayedItem, isGenerating]
  );

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [canCopyVideoFrame, setCanCopyVideoFrame] = useState(false);
  const dragSessionRef = useRef<ResizeDragSession | null>(null);

  const syncVideoCopyAvailability = useCallback(() => {
    setCanCopyVideoFrame(canCopyImage() && isVideoFrameReady(videoRef.current));
  }, []);

  const setVideoElementRef = useCallback((element: HTMLVideoElement | null) => {
    videoRef.current = element;
    if (element) {
      element.disablePictureInPicture = true;
    }
    setCanCopyVideoFrame(canCopyImage() && isVideoFrameReady(element));
  }, []);

  const handleCopyCurrentFrame = useCallback(
    async (event: MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
      const video = videoRef.current;
      if (!canCopyImage()) {
        notifications.error(t('widgets.preview.copyCurrentFrameUnsupported'));
        return;
      }
      if (!isVideoFrameReady(video)) {
        notifications.error(t('widgets.preview.copyCurrentFrameNotReady'));
        return;
      }
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        notifications.error(t('widgets.preview.copyCurrentFrameDrawFailed'));
        return;
      }
      try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await encodeCanvasPng(canvas);
        if (!blob) {
          notifications.error(t('widgets.preview.copyCurrentFrameEncodeFailed'));
          return;
        }
        await copyImage(blob);
        notifications.success(t('widgets.preview.copyCurrentFrameSuccess'));
      } catch {
        notifications.error(t('widgets.preview.copyCurrentFrameWriteFailed'));
      }
    },
    [notifications, t]
  );

  const resetSize = useCallback(
    (event: MouseEvent<HTMLElement> | KeyboardEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      setLiveSize(null);
      editGraph({ height: undefined, nodeId: node.id, type: 'setCurrentImageNodeSize', width: undefined });
    },
    [editGraph, node.id]
  );

  const handleResizePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      event.currentTarget.setPointerCapture?.(event.pointerId);
      const shell = event.currentTarget.closest<HTMLElement>('[data-workflow-node-shell="true"]');
      const rect = shell?.getBoundingClientRect();
      const zoom = rect && rect.width > 0 && savedWidth > 0 ? rect.width / savedWidth : 1;
      dragSessionRef.current = {
        pointerId: event.pointerId,
        startHeight: savedHeight,
        startWidth: savedWidth,
        startX: event.clientX,
        startY: event.clientY,
        zoom,
      };
    },
    [savedHeight, savedWidth]
  );

  const handleResizePointerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
    const session = dragSessionRef.current;
    if (!session || session.pointerId !== event.pointerId) {
      return;
    }
    const nextWidth = clampCurrentImageWidth(session.startWidth + (event.clientX - session.startX) / session.zoom);
    const nextHeight = clampCurrentImageHeight(session.startHeight + (event.clientY - session.startY) / session.zoom);
    setLiveSize({ height: nextHeight, width: nextWidth });
  }, []);

  const handleResizePointerUp = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const session = dragSessionRef.current;
      if (!session || session.pointerId !== event.pointerId) {
        return;
      }
      dragSessionRef.current = null;
      const nextWidth = clampCurrentImageWidth(session.startWidth + (event.clientX - session.startX) / session.zoom);
      const nextHeight = clampCurrentImageHeight(session.startHeight + (event.clientY - session.startY) / session.zoom);
      setLiveSize(null);
      if (nextWidth !== session.startWidth || nextHeight !== session.startHeight) {
        editGraph({ height: nextHeight, nodeId: node.id, type: 'setCurrentImageNodeSize', width: nextWidth });
      }
    },
    [editGraph, node.id]
  );

  const handleResizePointerCancel = useCallback(() => {
    dragSessionRef.current = null;
    setLiveSize(null);
  }, []);

  const handleResizeKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (
        event.key === 'Home' ||
        event.key === 'Backspace' ||
        event.key === 'Delete' ||
        event.key === 'Enter' ||
        event.key === ' '
      ) {
        resetSize(event);
        return;
      }
      const dx =
        event.key === 'ArrowRight' ? KEYBOARD_RESIZE_STEP_PX : event.key === 'ArrowLeft' ? -KEYBOARD_RESIZE_STEP_PX : 0;
      const dy =
        event.key === 'ArrowDown' ? KEYBOARD_RESIZE_STEP_PX : event.key === 'ArrowUp' ? -KEYBOARD_RESIZE_STEP_PX : 0;
      if (dx === 0 && dy === 0) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      const nextWidth = clampCurrentImageWidth(savedWidth + dx);
      const nextHeight = clampCurrentImageHeight(savedHeight + dy);
      if (nextWidth !== savedWidth || nextHeight !== savedHeight) {
        editGraph({ height: nextHeight, nodeId: node.id, type: 'setCurrentImageNodeSize', width: nextWidth });
      }
    },
    [editGraph, node.id, resetSize, savedHeight, savedWidth]
  );

  const fallbackImage = useMemo(
    () =>
      displayedItem && isGalleryImageItem(displayedItem)
        ? imageUrlToStreamingSource({
            alt: displayedItem.name,
            height: displayedItem.height,
            kind: 'fallback',
            src: displayedItem.fullUrl,
            width: displayedItem.width,
          })
        : null,
    [displayedItem]
  );
  const liveImage = useMemo(() => progressImageToStreamingSource(progressImage), [progressImage]);
  const videoPosterImage = useMemo(
    () =>
      displayedItem && displayedItem.kind === 'video'
        ? imageUrlToStreamingSource({
            alt: displayedItem.name,
            height: displayedItem.height,
            kind: 'fallback',
            src: displayedItem.thumbnailUrl,
            width: displayedItem.width,
          })
        : null,
    [displayedItem]
  );
  const videoPosterFallback = useMemo(
    () => <StreamingImageFrame fallbackImage={videoPosterImage} fit="contain" h="full" liveImage={null} w="full" />,
    [videoPosterImage]
  );

  const resizeLabel = t('widgets.workflow.currentImageResizeHandle', {
    height: displayHeight,
    width: displayWidth,
  });

  return (
    <Box
      data-is-selected={selected}
      data-workflow-node-shell="true"
      overflow="hidden"
      position="relative"
      w={`${displayWidth}px`}
      {...getWorkflowNodeShellProps({ selected })}
    >
      <Flex {...getWorkflowNodeHeaderProps()} gap="1.5">
        <MiddleTruncate
          flex="1"
          fontSize="lg"
          fontWeight="700"
          minW="0"
          text={node.data.label || t('widgets.workflow.addNodeDialog.currentImageTitle')}
        />
        <HStack className="nodrag nopan" flexShrink={0} gap="1" ms="auto">
          {isGenerating ? (
            <Text color="brand.solid" data-node-status-indicator="true" fontSize="xs" fontVariantNumeric="tabular-nums">
              {progressPercent !== null
                ? `${t('widgets.workflow.generating')} ${progressPercent}%`
                : t('widgets.workflow.generating')}
            </Text>
          ) : null}
          {isShowingVideo ? (
            <Tooltip content={canCopyImage() ? t('widgets.preview.copyCurrentFrame') : t('common.copyUnavailable')}>
              <IconButton
                aria-label={t('widgets.preview.copyCurrentFrame')}
                color="fg.muted"
                data-testid="current-image-copy-video-frame"
                disabled={!canCopyVideoFrame}
                size="xs"
                variant="ghost"
                onClick={handleCopyCurrentFrame}
              >
                <Icon as={CopyIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
          ) : null}
          {isCustomSize ? (
            <Tooltip content={resizeLabel}>
              <Button
                aria-label={resizeLabel}
                color="fg.muted"
                data-testid="current-image-node-reset-size"
                fontSize="xs"
                fontVariantNumeric="tabular-nums"
                h="5"
                minW="0"
                px="1.5"
                size="xs"
                variant="subtle"
                onClick={resetSize}
              >
                {`${displayWidth}×${displayHeight} ↺`}
              </Button>
            </Tooltip>
          ) : null}
        </HStack>
      </Flex>

      {isGenerating ? (
        <Box bg="bg.muted" data-node-progress-strip="true" h="2px" overflow="hidden" position="relative" w="full">
          <Box
            animationDuration={progressPercent === null ? 'var(--wb-motion-duration-slow)' : undefined}
            animationIterationCount={
              progressPercent === null ? 'var(--wb-motion-animation-iteration-count)' : undefined
            }
            animationName={progressPercent === null ? 'pulse' : undefined}
            bg="brand.solid"
            h="full"
            transition="width var(--wb-motion-duration-slow) ease"
            w={progressPercent === null ? 'full' : `${progressPercent}%`}
          />
        </Box>
      ) : null}

      <Box h={`${displayHeight}px`} overflow="hidden" position="relative" w="full">
        {isShowingVideo && displayedItem.kind === 'video' ? (
          <Box bg="black" className="nodrag nopan nowheel" h="full" position="relative" w="full">
            <Suspense fallback={videoPosterFallback}>
              <PlayerComponent
                ref={setVideoElementRef}
                aria-label={displayedItem.name}
                defaultLoop
                fps={displayedItem.fps}
                poster={displayedItem.thumbnailUrl}
                size="full"
                src={displayedItem.fullUrl}
                onCanPlay={syncVideoCopyAvailability}
                onLoadedData={syncVideoCopyAvailability}
                onLoadedMetadata={syncVideoCopyAvailability}
                onPlaying={syncVideoCopyAvailability}
                onResize={syncVideoCopyAvailability}
                onSeeked={syncVideoCopyAvailability}
              />
            </Suspense>
          </Box>
        ) : (
          <StreamingImageFrame fallbackImage={fallbackImage} fit="contain" h="full" liveImage={liveImage} w="full">
            <Flex align="center" color="fg.subtle" fontSize="xs" h="full" justify="center" px="4" textAlign="center">
              {t('widgets.workflow.currentImageEmpty')}
            </Flex>
          </StreamingImageFrame>
        )}

        {dragRef ? <CurrentMediaDragGrip itemRef={dragRef} nodeId={node.id} /> : null}

        {isGenerating ? (
          <Flex
            align="center"
            aria-hidden="true"
            bg="bg/85"
            bottom="2"
            color="fg"
            data-testid="current-image-progress-badge"
            gap="1.5"
            left="2"
            pointerEvents="none"
            position="absolute"
            px={progressPercent !== null ? '2' : '1'}
            py="1"
            rounded="full"
            shadow="sm"
            zIndex="1"
          >
            <Box
              animationDuration={progressPercent === null ? '0.9s' : undefined}
              animationIterationCount={progressPercent === null ? 'infinite' : undefined}
              animationName={progressPercent === null ? 'spin' : undefined}
              animationTimingFunction={progressPercent === null ? 'linear' : undefined}
              boxSize="3.5"
            >
              <svg height="100%" viewBox="0 0 16 16" width="100%">
                <circle cx="8" cy="8" fill="none" r="6" stroke="currentColor" strokeOpacity="0.28" strokeWidth="2" />
                <circle
                  cx="8"
                  cy="8"
                  fill="none"
                  r="6"
                  stroke="currentColor"
                  strokeDasharray={RING_CIRCUMFERENCE}
                  strokeDashoffset={
                    progressPercent === null
                      ? INDETERMINATE_DASH_OFFSET
                      : RING_CIRCUMFERENCE * (1 - progressPercent / 100)
                  }
                  strokeLinecap="round"
                  strokeWidth="2"
                  transform="rotate(-90 8 8)"
                />
              </svg>
            </Box>
            {progressPercent !== null ? (
              <Text fontSize="xs" fontVariantNumeric="tabular-nums" fontWeight="600">
                {`${progressPercent}%`}
              </Text>
            ) : null}
          </Flex>
        ) : null}
      </Box>

      {/* Dedicated footer strip keeps the resize grip completely clear of VideoPlayer's Fullscreen control. */}
      <Flex align="center" bg="bg.subtle" borderTopWidth="1px" h="3.5" justify="end" px="1" w="full">
        <Box
          _focusVisible={RESIZE_FOCUS_VISIBLE_STYLE}
          _hover={RESIZE_HOVER_STYLE}
          aria-label={resizeLabel}
          boxSize="3.5"
          className="nodrag nopan"
          cursor="nwse-resize"
          data-testid="current-image-node-resize-handle"
          opacity={selected ? 0.9 : 0.55}
          position="relative"
          role="button"
          tabIndex={0}
          onDoubleClick={resetSize}
          onKeyDown={handleResizeKeyDown}
          onLostPointerCapture={handleResizePointerCancel}
          onPointerCancel={handleResizePointerCancel}
          onPointerDown={handleResizePointerDown}
          onPointerMove={handleResizePointerMove}
          onPointerUp={handleResizePointerUp}
        >
          <Box
            borderBottomWidth="2px"
            borderColor="fg.subtle"
            borderRightWidth="2px"
            bottom="0.5"
            boxSize="2"
            pointerEvents="none"
            position="absolute"
            right="0.5"
          />
        </Box>
      </Flex>
    </Box>
  );
};

const CurrentMediaDragGrip = ({ itemRef, nodeId }: { itemRef: GalleryItemRef; nodeId: string }) => {
  const { t } = useTranslation();
  const dragData = useMemo(() => getGalleryItemDragData([itemRef]), [itemRef]);
  const { isDragging, listeners, setNodeRef } = useDraggable({
    data: dragData,
    id: getGalleryItemDragId(itemRef, 'preview-frame', `current-image-${nodeId}`),
  });
  const label = itemRef.kind === 'video' ? t('widgets.preview.dragVideo') : t('widgets.workflow.currentImageDragImage');

  return (
    <Tooltip content={label}>
      <Badge
        ref={setNodeRef}
        {...listeners}
        className="nodrag nopan"
        cursor={isDragging ? 'grabbing' : 'grab'}
        data-testid="current-image-media-drag-grip"
        insetInlineStart="50%"
        opacity={isDragging ? 0.55 : 0.82}
        position="absolute"
        top="2"
        touchAction="none"
        transform="translateX(-50%)"
        variant="solid"
        zIndex="2"
      >
        <GripHorizontalIcon aria-hidden="true" size={12} />
      </Badge>
    </Tooltip>
  );
};

const CurrentImageSnapshotNode = ({ data, selected }: NodeProps<CurrentImageFlowNodeType>) => {
  const { t } = useTranslation();
  const node = data.documentNode;
  const width = node.data.width ?? CURRENT_IMAGE_DEFAULT_WIDTH;

  return (
    <Box
      data-is-selected={selected}
      data-workflow-node-shell="true"
      w={`${width}px`}
      {...getWorkflowNodeShellProps({ selected })}
    >
      <Flex {...getWorkflowNodeHeaderProps()}>
        <MiddleTruncate
          data-workflow-export-node-title="true"
          data-workflow-export-static-node-content="true"
          flex="1"
          fontSize="lg"
          fontWeight="700"
          minW="0"
          text={node.data.label || t('widgets.workflow.addNodeDialog.currentImageTitle')}
        />
      </Flex>
    </Box>
  );
};

const CurrentImageFlowNodeComponent = (props: NodeProps<CurrentImageFlowNodeType>) => {
  const isWorkflowImageExport = useIsWorkflowImageExport();

  return isWorkflowImageExport ? <CurrentImageSnapshotNode {...props} /> : <CurrentImageEditorNode {...props} />;
};

export const CurrentImageFlowNode = memo(CurrentImageFlowNodeComponent);

import type {
  GalleryItem,
  GalleryItemRef,
  GalleryVideoItem,
  GeneratedImageContract,
} from '@features/gallery/contracts';
import type { QueueActiveSession } from '@features/queue/contracts';
import type * as VideoPlayerModule from '@platform/ui/video-player/VideoPlayer';
import type { NodeProps } from '@xyflow/react';

import { Badge, Box, Flex, HStack, Icon, ProgressCircle, Spinner, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import {
  getSelectedGalleryItemFromValues,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  toGalleryItemKey,
  toGalleryItemRef,
} from '@features/gallery/contracts';
import { useGalleryItemContextMenu } from '@features/gallery/react';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { getAuthSession, refreshProtectedMediaCookie } from '@features/identity';
import { getAutoSwitchFollowedLive, getQueueActiveSessions } from '@features/queue/contracts';
import {
  useActiveProgressTargets,
  useFollowedProgressTargets,
  useFramedProgressTargets,
  useItemProgress,
  useQueueItemProgress,
  useQueueItemProgressImage,
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
  WorkflowNodeHeaderButton,
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
import { CopyIcon, EllipsisVerticalIcon, GripHorizontalIcon } from 'lucide-react';
import {
  lazy,
  memo,
  Suspense,
  useCallback,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
} from 'react';
import { useTranslation } from 'react-i18next';

import type { CurrentImageFlowNode as CurrentImageFlowNodeType } from './flowAdapters';

const KEYBOARD_RESIZE_STEP_PX = 32;

const RESIZE_FOCUS_VISIBLE_STYLE = {
  opacity: 1,
  outline: '2px solid',
  outlineColor: 'accent.solid',
  outlineOffset: '-2px',
} as const;
const RESIZE_HOVER_STYLE = { opacity: 1 } as const;
const VIDEO_FAILURE_POSTER_STYLE: CSSProperties = {
  display: 'block',
  height: '100%',
  inset: 0,
  objectFit: 'contain',
  position: 'absolute',
  width: '100%',
};

// Load VideoPlayer and its frame/recovery helpers together only when a video is displayed, keeping initial routes light.
let loadedVideoPlayer: typeof VideoPlayerModule | null = null;
const LazyVideoPlayer = lazy(() =>
  import('@platform/ui/video-player/VideoPlayer').then((mod) => {
    loadedVideoPlayer = mod;
    return { default: mod.VideoPlayer };
  })
);

// Keep VideoPlayer typed while still splitting the player chunk until a video is selected.
const PlayerComponent: typeof VideoPlayerModule.VideoPlayer =
  LazyVideoPlayer as unknown as typeof VideoPlayerModule.VideoPlayer;

interface ResizeDragSession {
  nodeId: string;
  pointerId: number;
  projectId: string;
  startHeight: number;
  startWidth: number;
  startX: number;
  startY: number;
  zoom: number;
}

interface DisplayedGalleryMedia {
  item: GalleryItem;
  /** Full selected GalleryItem opens the menu without refetching; legacy recentImages fall back to a ref. */
  menuTarget: GalleryItem | GalleryItemRef;
}

export interface CurrentImageExecutionState {
  dragRef: GalleryItemRef | null;
  hasLiveFrame: boolean;
  isGenerating: boolean;
  isShowingVideo: boolean;
  isWaiting: boolean;
  progressPercent: number | null;
}

/**
 * Pick the displayed live session using the queue feature's shared start-order policy (`getAutoSwitchFollowedLive`),
 * avoiding per-frame flipping across concurrent multi-GPU slots while keeping a running unframed slot tracked for
 * progress before its first frame arrives.
 */
export const resolveCurrentImageActiveSession = (
  sessions: readonly QueueActiveSession[]
): { liveSession: QueueActiveSession | null; trackedSession: QueueActiveSession | null } => {
  const liveSession = getAutoSwitchFollowedLive('progress', sessions, null);
  const trackedSession =
    liveSession ?? sessions.filter((session) => session.state === 'running').at(-1) ?? sessions[0] ?? null;
  return { liveSession, trackedSession };
};

// Prefer the coordinator's percentage when its active slot matches the displayed session; use per-slot lookup otherwise.
export const resolveProgressPercentage = ({
  activeProgress,
  activeTarget,
  slotProgressPercentage,
}: {
  activeProgress: { activeItemIndex?: number; percentage: number | null } | null | undefined;
  activeTarget: { itemIndex: number; queueItemId: string } | null;
  slotProgressPercentage?: number | null;
}): number | null => {
  if (activeTarget && activeProgress && activeProgress.activeItemIndex === activeTarget.itemIndex) {
    return activeProgress.percentage;
  }
  return slotProgressPercentage ?? null;
};

// Keep queued workflow runs in Waiting until a session owned by this project is actively executing.
export const resolveCurrentImageExecutionState = ({
  displayedItem,
  hasRunningSession,
  isWorkflowRunning,
  progressImage,
  progressPercentage,
}: {
  displayedItem: GalleryItem | null;
  hasRunningSession: boolean;
  isWorkflowRunning: boolean;
  progressImage: unknown;
  progressPercentage: number | null | undefined;
}): CurrentImageExecutionState => {
  const hasLiveFrame = Boolean(progressImage);
  const isGenerating = hasLiveFrame || hasRunningSession;
  const isWaiting = !isGenerating && isWorkflowRunning;
  const isShowingVideo = !hasLiveFrame && displayedItem?.kind === 'video';
  const dragRef = !hasLiveFrame && displayedItem ? toGalleryItemRef(displayedItem) : null;
  const progressPercent =
    isGenerating && typeof progressPercentage === 'number' && progressPercentage > 0
      ? Math.round(Math.min(100, Math.max(0, progressPercentage * 100)))
      : null;

  return { dragRef, hasLiveFrame, isGenerating, isShowingVideo, isWaiting, progressPercent };
};

const isFullGalleryItem = (media: GalleryItem | GalleryItemRef): media is GalleryItem => 'fullUrl' in media;

const resolveDisplayedGalleryMedia = (values: Record<string, unknown>): DisplayedGalleryMedia | null => {
  const selected = getSelectedGalleryItemFromValues(values);
  if (selected) {
    return { item: selected, menuTarget: selected };
  }

  const recentImages = Array.isArray(values.recentImages) ? (values.recentImages as GeneratedImageContract[]) : [];
  const latest = recentImages[0];
  if (!latest) {
    return null;
  }

  const item = legacyGeneratedImageToGalleryItem(latest);
  return { item, menuTarget: toGalleryItemRef(item) };
};

const isSameDisplayedGalleryItem = (a: GalleryItem | null, b: GalleryItem | null): boolean => {
  if (a === b) {
    return true;
  }
  if (a === null || b === null) {
    return false;
  }
  const aRecord = a as unknown as Record<string, unknown>;
  const bRecord = b as unknown as Record<string, unknown>;
  const aKeys = Object.keys(aRecord);
  const bKeys = Object.keys(bRecord);
  return aKeys.length === bKeys.length && aKeys.every((key) => Object.is(aRecord[key], bRecord[key]));
};

const isSameDisplayedGalleryMedia = (a: DisplayedGalleryMedia | null, b: DisplayedGalleryMedia | null): boolean =>
  a === b ||
  (a !== null &&
    b !== null &&
    isFullGalleryItem(a.menuTarget) === isFullGalleryItem(b.menuTarget) &&
    isSameDisplayedGalleryItem(a.item, b.item));

const isTargetInFullscreen = (target: EventTarget | null): boolean =>
  target instanceof Node && Boolean(target.ownerDocument?.fullscreenElement?.contains(target));

const CurrentImageEditorNode = ({ data, selected }: NodeProps<CurrentImageFlowNodeType>) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const notifications = useWorkflowNotifications();
  // Select only the shown media item so unrelated gallery changes (board filter, page, scroll) do not re-render.
  const displayedMedia = useWorkflowProjectSelector(
    (project) => resolveDisplayedGalleryMedia(project.galleryValues),
    isSameDisplayedGalleryMedia
  );
  const displayedItem = displayedMedia?.item ?? null;
  const projectId = useWorkflowProjectSelector((project) => project.id);
  const isWorkflowRunning = useWorkflowProjectSelector((project) => project.isWorkflowRunning);
  const queueItems = useWorkflowProjectSelector((project) => project.queueItems);

  const activeTargets = useActiveProgressTargets();
  const followedTargets = useFollowedProgressTargets();
  const framedTargets = useFramedProgressTargets();

  const sessions = useMemo(
    () => getQueueActiveSessions(queueItems, activeTargets, followedTargets, framedTargets),
    [queueItems, activeTargets, followedTargets, framedTargets]
  );
  const { liveSession, trackedSession } = useMemo(() => resolveCurrentImageActiveSession(sessions), [sessions]);
  const hasRunningSession = useMemo(() => sessions.some((session) => session.state === 'running'), [sessions]);

  const slotProgressImage = useQueueItemProgressImage(liveSession?.queueItemId ?? '', liveSession?.itemIndex ?? 0);
  const projectProgressImage = liveSession ? slotProgressImage : null;
  const activeProgress = useQueueItemProgress(trackedSession?.queueItemId ?? '');
  const slotProgress = useItemProgress(trackedSession?.backendItemId ?? null);
  const progressPercentage = useMemo(
    () =>
      resolveProgressPercentage({
        activeProgress,
        activeTarget: trackedSession,
        slotProgressPercentage: slotProgress?.percentage,
      }),
    [activeProgress, trackedSession, slotProgress?.percentage]
  );

  const displayedItemKey = useMemo(() => (displayedItem ? toGalleryItemKey(displayedItem) : null), [displayedItem]);
  const { hasLiveFrame, isGenerating, isShowingVideo, isWaiting, progressPercent } = useMemo(
    () =>
      resolveCurrentImageExecutionState({
        displayedItem,
        hasRunningSession,
        isWorkflowRunning,
        progressImage: projectProgressImage,
        progressPercentage,
      }),
    [displayedItem, hasRunningSession, isWorkflowRunning, progressPercentage, projectProgressImage]
  );
  const videoDragRef = useMemo<GalleryItemRef | null>(
    () => (!hasLiveFrame && displayedItem?.kind === 'video' ? toGalleryItemRef(displayedItem) : null),
    [displayedItem, hasLiveFrame]
  );
  const videoSurfaceKey =
    displayedItem?.kind === 'video' && displayedItemKey ? `${displayedItemKey}:${displayedItem.fullUrl}` : null;

  const node = data.documentNode;
  const savedWidth = clampCurrentImageWidth(node.data.width);
  const savedHeight = clampCurrentImageHeight(node.data.height);
  const [liveSizeState, setLiveSizeState] = useState<{
    height: number;
    nodeId: string;
    projectId: string;
    width: number;
  } | null>(null);
  const liveSize =
    liveSizeState && liveSizeState.nodeId === node.id && liveSizeState.projectId === projectId ? liveSizeState : null;
  const displayWidth = liveSize?.width ?? savedWidth;
  const displayHeight = liveSize?.height ?? savedHeight;
  const isCustomSize = displayWidth !== CURRENT_IMAGE_DEFAULT_WIDTH || displayHeight !== CURRENT_IMAGE_DEFAULT_HEIGHT;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const resizeHandleRef = useRef<HTMLDivElement | null>(null);
  const [copyReadyVideoKey, setCopyReadyVideoKey] = useState<string | null>(null);
  const canCopyVideoFrame = isShowingVideo && videoSurfaceKey !== null && copyReadyVideoKey === videoSurfaceKey;
  const dragSessionRef = useRef<ResizeDragSession | null>(null);

  const handleVideoElementChange = useCallback((element: HTMLVideoElement | null) => {
    videoRef.current = element;
  }, []);

  const handleVideoCopyAvailabilityChange = useCallback((surfaceKey: string, isAvailable: boolean) => {
    setCopyReadyVideoKey((prev) => (isAvailable ? surfaceKey : prev === surfaceKey ? null : prev));
  }, []);

  const handleCopyCurrentFrame = useCallback(
    async (event: MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
      const video = videoRef.current;
      const playerUtils = loadedVideoPlayer;
      if (!canCopyImage()) {
        notifications.error(t('widgets.preview.copyCurrentFrameUnsupported'));
        return;
      }
      if (!playerUtils || !playerUtils.isVideoFrameReady(video)) {
        notifications.error(t('widgets.preview.copyCurrentFrameNotReady'));
        return;
      }

      const accountEpoch = getAuthSession().accountEpoch;
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
      } catch {
        notifications.error(t('widgets.preview.copyCurrentFrameDrawFailed'));
        return;
      }

      let blob: Blob | null;
      try {
        blob = await playerUtils.encodeCanvasPng(canvas);
      } catch (error: unknown) {
        notifications.error(
          t(
            playerUtils.isCanvasSecurityError(error)
              ? 'widgets.preview.copyCurrentFrameDrawFailed'
              : 'widgets.preview.copyCurrentFrameEncodeFailed'
          )
        );
        return;
      }
      if (!blob) {
        notifications.error(t('widgets.preview.copyCurrentFrameEncodeFailed'));
        return;
      }

      const isCurrent = (): boolean => videoRef.current === video && getAuthSession().accountEpoch === accountEpoch;
      if (!isCurrent()) {
        notifications.error(t('widgets.preview.copyCurrentFrameStale'));
        return;
      }

      try {
        await copyImage(blob);
      } catch {
        notifications.error(t('widgets.preview.copyCurrentFrameWriteFailed'));
        return;
      }

      if (!isCurrent()) {
        notifications.error(t('widgets.preview.copyCurrentFrameStale'));
        return;
      }
      notifications.success(t('widgets.preview.copyCurrentFrameSuccess'));
    },
    [notifications, t]
  );

  const resetSize = useCallback(
    (event: MouseEvent<HTMLElement> | KeyboardEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      dragSessionRef.current = null;
      setLiveSizeState(null);
      editGraph({ height: undefined, nodeId: node.id, type: 'setCurrentImageNodeSize', width: undefined });
      resizeHandleRef.current?.focus();
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
        nodeId: node.id,
        pointerId: event.pointerId,
        projectId,
        startHeight: savedHeight,
        startWidth: savedWidth,
        startX: event.clientX,
        startY: event.clientY,
        zoom,
      };
    },
    [node.id, projectId, savedHeight, savedWidth]
  );

  const handleResizePointerMove = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const session = dragSessionRef.current;
      if (!session || session.pointerId !== event.pointerId) {
        return;
      }
      if (session.nodeId !== node.id || session.projectId !== projectId) {
        dragSessionRef.current = null;
        setLiveSizeState(null);
        return;
      }
      const nextWidth = clampCurrentImageWidth(session.startWidth + (event.clientX - session.startX) / session.zoom);
      const nextHeight = clampCurrentImageHeight(session.startHeight + (event.clientY - session.startY) / session.zoom);
      setLiveSizeState({ height: nextHeight, nodeId: node.id, projectId, width: nextWidth });
    },
    [node.id, projectId]
  );

  const handleResizePointerUp = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const session = dragSessionRef.current;
      if (!session || session.pointerId !== event.pointerId) {
        return;
      }
      dragSessionRef.current = null;
      setLiveSizeState(null);
      if (session.nodeId !== node.id || session.projectId !== projectId) {
        return;
      }
      const nextWidth = clampCurrentImageWidth(session.startWidth + (event.clientX - session.startX) / session.zoom);
      const nextHeight = clampCurrentImageHeight(session.startHeight + (event.clientY - session.startY) / session.zoom);
      if (nextWidth !== session.startWidth || nextHeight !== session.startHeight) {
        editGraph({ height: nextHeight, nodeId: node.id, type: 'setCurrentImageNodeSize', width: nextWidth });
      }
    },
    [editGraph, node.id, projectId]
  );

  const handleResizePointerCancel = useCallback(() => {
    dragSessionRef.current = null;
    setLiveSizeState(null);
  }, []);

  const handleResizeKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'Home' || event.key === 'Enter' || event.key === ' ') {
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

  const title = node.data.label || t('widgets.workflow.addNodeDialog.currentImageTitle');
  const shownMediaRef = useMemo<GalleryItemRef | null>(
    () => (!hasLiveFrame && displayedItem ? toGalleryItemRef(displayedItem) : null),
    [displayedItem, hasLiveFrame]
  );
  const draggableImageRef = shownMediaRef?.kind === 'image' ? shownMediaRef : null;
  const shownMediaMenuTarget = !hasLiveFrame && displayedMedia ? displayedMedia.menuTarget : null;
  const openItemMenu = useGalleryItemContextMenu(shownMediaMenuTarget);
  const handleMediaContextMenu = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      // The gallery menu portals under document.body, which sits behind a fullscreen top-layer video.
      // Stop propagation without preventDefault so ReactFlow's node menu stays closed and the browser menu opens.
      if (isTargetInFullscreen(event.target)) {
        event.stopPropagation();
        return;
      }
      openItemMenu?.(event);
    },
    [openItemMenu]
  );
  const imageActionsLabel = t('widgets.preview.imageActions');
  const fallbackImage = useMemo(
    () =>
      displayedItem && isGalleryImageItem(displayedItem)
        ? imageUrlToStreamingSource({
            alt: displayedItem.name || title,
            height: displayedItem.height,
            kind: 'fallback',
            src: displayedItem.fullUrl,
            width: displayedItem.width,
          })
        : null,
    [displayedItem, title]
  );
  const liveImage = useMemo(() => progressImageToStreamingSource(projectProgressImage), [projectProgressImage]);

  const resizeLabel = t('widgets.workflow.currentImageResizeHandle', {
    height: displayHeight,
    width: displayWidth,
  });
  const resetSizeLabel = t('widgets.workflow.currentImageResetSize', {
    height: displayHeight,
    width: displayWidth,
  });
  const showItemMenuButton = Boolean(openItemMenu);

  return (
    <Box
      data-is-selected={selected}
      data-workflow-node-shell="true"
      display="flex"
      flexDirection="column"
      h={`${displayHeight}px`}
      overflow="hidden"
      position="relative"
      w={`${displayWidth}px`}
      {...getWorkflowNodeShellProps({ selected })}
    >
      <Flex {...getWorkflowNodeHeaderProps({ endsWithButton: showItemMenuButton })} flexShrink={0} gap="1.5">
        <MiddleTruncate flex="1" fontWeight="600" minW="0" text={title} />
        <HStack className="nodrag nopan" flexShrink={0} gap="1" ms="auto">
          {isGenerating ? (
            // Keep header compact at 200 px min-width; exact percentage is already displayed in the progress badge.
            <Text color="brand.solid" data-node-status-indicator="true" fontSize="xs">
              {t('widgets.workflow.generating')}
            </Text>
          ) : isWaiting ? (
            <Text color="fg.muted" data-node-status-indicator="true" fontSize="xs">
              {t('common.status.waiting')}
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
            <Tooltip content={resetSizeLabel}>
              <Button
                aria-label={resetSizeLabel}
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
          {showItemMenuButton ? (
            <WorkflowNodeHeaderButton
              aria-haspopup="menu"
              aria-label={imageActionsLabel}
              content={imageActionsLabel}
              onClick={openItemMenu}
            >
              <Icon as={EllipsisVerticalIcon} boxSize="3.5" />
            </WorkflowNodeHeaderButton>
          ) : null}
        </HStack>
      </Flex>

      {isGenerating ? (
        <Box
          bg="bg.muted"
          data-node-progress-strip="true"
          flexShrink={0}
          h="2px"
          overflow="hidden"
          position="relative"
          w="full"
        >
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

      <Box
        flex="1"
        minH="0"
        overflow="hidden"
        position="relative"
        w="full"
        onContextMenu={openItemMenu || isShowingVideo ? handleMediaContextMenu : undefined}
      >
        {isShowingVideo && displayedItem?.kind === 'video' && videoSurfaceKey ? (
          <CurrentImageVideoSurface
            key={videoSurfaceKey}
            item={displayedItem}
            surfaceKey={videoSurfaceKey}
            onCopyAvailabilityChange={handleVideoCopyAvailabilityChange}
            onVideoElementChange={handleVideoElementChange}
          />
        ) : (
          <DraggableCurrentImage itemRef={draggableImageRef}>
            <StreamingImageFrame fallbackImage={fallbackImage} fit="contain" h="full" liveImage={liveImage} w="full">
              <Flex align="center" color="fg.subtle" fontSize="xs" h="full" justify="center" px="4" textAlign="center">
                {t('widgets.workflow.currentImageEmpty')}
              </Flex>
            </StreamingImageFrame>
          </DraggableCurrentImage>
        )}

        {isShowingVideo && videoDragRef ? <CurrentMediaDragGrip itemRef={videoDragRef} nodeId={node.id} /> : null}

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
            {progressPercent === null ? (
              <Spinner borderWidth="2px" boxSize="3.5" />
            ) : (
              <ProgressCircle.Root size="xs" value={progressPercent}>
                <ProgressCircle.Circle>
                  <ProgressCircle.Track />
                  <ProgressCircle.Range stroke="currentColor" strokeLinecap="round" />
                </ProgressCircle.Circle>
              </ProgressCircle.Root>
            )}
            {progressPercent !== null ? (
              <Text fontSize="xs" fontVariantNumeric="tabular-nums" fontWeight="600">
                {`${progressPercent}%`}
              </Text>
            ) : null}
          </Flex>
        ) : null}
      </Box>

      {/* Dedicated footer strip keeps the resize grip clear of VideoPlayer's Fullscreen control. */}
      <Flex align="center" bg="bg.subtle" borderTopWidth="1px" flexShrink={0} h="3.5" justify="end" px="1" w="full">
        <Box
          ref={resizeHandleRef}
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

/** The image under the node's frame, draggable wherever gallery images can be dropped. */
const DraggableCurrentImage = ({ children, itemRef }: { children: ReactNode; itemRef: GalleryItemRef | null }) => {
  const disabledId = useId();
  const dragData = useMemo(() => (itemRef ? getGalleryItemDragData([itemRef]) : undefined), [itemRef]);
  // Pointer drags only, as the Preview frame's: the image is no tab stop, so the canvas keeps its own keys. The header's
  // button is the keyboard's way to the item menu.
  const { listeners, setNodeRef } = useDraggable({
    data: dragData,
    disabled: !itemRef,
    id: itemRef
      ? getGalleryItemDragId(itemRef, 'workflow-current-image', disabledId)
      : `workflow-current-image:disabled:${disabledId}`,
  });

  return (
    // `nodrag`: the canvas leaves the pointer to the image drag rather than moving the node.
    <Box ref={setNodeRef} className="nodrag" cursor={itemRef ? 'grab' : undefined} h="full" w="full" {...listeners}>
      {children}
    </Box>
  );
};

const CurrentImageVideoSurface = ({
  item,
  onCopyAvailabilityChange,
  onVideoElementChange,
  surfaceKey,
}: {
  item: GalleryVideoItem;
  onCopyAvailabilityChange: (surfaceKey: string, isAvailable: boolean) => void;
  onVideoElementChange: (element: HTMLVideoElement | null) => void;
  surfaceKey: string;
}) => {
  const { t } = useTranslation();
  const localVideoRef = useRef<HTMLVideoElement | null>(null);
  const [hasFailed, setHasFailed] = useState(false);

  const publishCopyAvailability = useCallback(() => {
    onCopyAvailabilityChange(surfaceKey, Boolean(loadedVideoPlayer?.isVideoFrameCopyAvailable(localVideoRef.current)));
  }, [onCopyAvailabilityChange, surfaceKey]);

  const setVideoElementRef = useCallback(
    (element: HTMLVideoElement | null) => {
      localVideoRef.current = element;
      onVideoElementChange(element);
      if (element) {
        element.disablePictureInPicture = true;
      }
      onCopyAvailabilityChange(surfaceKey, Boolean(loadedVideoPlayer?.isVideoFrameCopyAvailable(element)));
    },
    [onCopyAvailabilityChange, onVideoElementChange, surfaceKey]
  );

  const recoveryRef = useRef<VideoPlayerModule.VideoErrorRecovery | null>(null);
  const getRecovery = useCallback(() => {
    if (!recoveryRef.current && loadedVideoPlayer) {
      recoveryRef.current = loadedVideoPlayer.createVideoErrorRecovery({
        getAccountEpoch: () => getAuthSession().accountEpoch,
        onFailedChange: setHasFailed,
        refreshCookie: refreshProtectedMediaCookie,
      });
    }
    return recoveryRef.current;
  }, []);

  const handleVideoError = useCallback(() => {
    publishCopyAvailability();
    const video = localVideoRef.current;
    getRecovery()?.handleError(video, () => localVideoRef.current === video);
  }, [getRecovery, publishCopyAvailability]);

  const handleLoadedMetadata = useCallback(() => {
    publishCopyAvailability();
    const video = localVideoRef.current;
    if (video && video.paused && video.currentTime === 0 && loadedVideoPlayer) {
      loadedVideoPlayer.seekVideo(video, loadedVideoPlayer.FIRST_FRAME_SEEK_SECONDS);
    }
  }, [publishCopyAvailability]);

  const handleRetryClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
      getRecovery()?.retry(localVideoRef.current);
      publishCopyAvailability();
    },
    [getRecovery, publishCopyAvailability]
  );

  const videoPosterImage = useMemo(
    () =>
      imageUrlToStreamingSource({
        alt: item.name,
        height: item.height,
        kind: 'fallback',
        src: item.thumbnailUrl,
        width: item.width,
      }),
    [item.height, item.name, item.thumbnailUrl, item.width]
  );
  const videoPosterFallback = useMemo(
    () => <StreamingImageFrame fallbackImage={videoPosterImage} fit="contain" h="full" liveImage={null} w="full" />,
    [videoPosterImage]
  );

  const failureOverlay = useMemo(
    () =>
      hasFailed ? (
        <>
          <img
            aria-hidden="true"
            alt=""
            draggable={false}
            height={item.height}
            src={item.thumbnailUrl}
            style={VIDEO_FAILURE_POSTER_STYLE}
            width={item.width}
          />
          <Flex
            align="center"
            bg="blackAlpha.700"
            data-testid="current-image-video-failure"
            direction="column"
            gap="2.5"
            inset="0"
            justify="center"
            position="absolute"
            px="3"
            textAlign="center"
          >
            <Text color="white" fontSize="sm" fontWeight="semibold">
              {t('widgets.preview.videoFailed')}
            </Text>
            <Button aria-label={t('widgets.preview.videoRetry')} size="sm" onClick={handleRetryClick}>
              {t('widgets.preview.videoRetry')}
            </Button>
          </Flex>
        </>
      ) : null,
    [handleRetryClick, hasFailed, item.height, item.thumbnailUrl, item.width, t]
  );

  return (
    <Box bg="black" className="nodrag nopan nowheel" h="full" position="relative" w="full">
      <Suspense fallback={videoPosterFallback}>
        <PlayerComponent
          ref={setVideoElementRef}
          aria-label={item.name}
          defaultLoop
          fps={item.fps}
          overlay={failureOverlay}
          poster={item.thumbnailUrl}
          size="full"
          src={item.fullUrl}
          onCanPlay={publishCopyAvailability}
          onEmptied={publishCopyAvailability}
          onError={handleVideoError}
          onLoadedData={publishCopyAvailability}
          onLoadedMetadata={handleLoadedMetadata}
          onPlaying={publishCopyAvailability}
          onResize={publishCopyAvailability}
          onSeeked={publishCopyAvailability}
          onWaiting={publishCopyAvailability}
        />
      </Suspense>
    </Box>
  );
};

const CurrentMediaDragGrip = ({ itemRef, nodeId }: { itemRef: GalleryItemRef; nodeId: string }) => {
  const { t } = useTranslation();
  const dragData = useMemo(() => getGalleryItemDragData([itemRef]), [itemRef]);
  const { isDragging, listeners, setNodeRef } = useDraggable({
    data: dragData,
    id: getGalleryItemDragId(itemRef, 'workflow-current-image', nodeId),
  });
  const label = t('widgets.preview.dragVideo');

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
  const width = clampCurrentImageWidth(node.data.width);

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
          fontWeight="600"
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

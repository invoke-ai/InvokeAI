import type { GalleryItem, GalleryItemRef, GeneratedImageContract } from '@features/gallery/contracts';
import type { QueueActiveSession, QueueItemProgress } from '@features/queue/contracts';
import type { NodeProps } from '@xyflow/react';

import { Box, Flex, HStack, Icon, ProgressCircle, Spinner, Text } from '@chakra-ui/react';
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
  CURRENT_IMAGE_MAX_SIZE,
  CURRENT_IMAGE_MIN_HEIGHT,
  CURRENT_IMAGE_MIN_WIDTH,
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
import { useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { canCopyImage } from '@platform/browser/clipboard';
import { Button } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ResizeCorner, trackResizeDrag, usePointerDrag } from '@platform/ui/ResizeHandle';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import {
  imageUrlToStreamingSource,
  progressImageToStreamingSource,
} from '@platform/ui/streaming-image/streamingImageSource';
import { Tooltip } from '@platform/ui/Tooltip';
import { useStoreApi } from '@xyflow/react';
import { CopyIcon, EllipsisVerticalIcon, RotateCcwIcon } from 'lucide-react';
import {
  lazy,
  memo,
  Suspense,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
} from 'react';
import { useTranslation } from 'react-i18next';

import type { CurrentImageVideoController } from './CurrentImageVideoSurface';
import type { CurrentImageFlowNode as CurrentImageFlowNodeType } from './flowAdapters';

const KEYBOARD_RESIZE_STEP_PX = 32;
/** Ignore sub-3px pointer jitter (such as the first click of a double-click) before committing a resize. */
const RESIZE_DRAG_THRESHOLD_PX = 3;
/** Below this width, collapse the dimension pill to an icon so the node title keeps readable room at 200 px. */
const COMPACT_HEADER_WIDTH = 280;
/** Switch VideoPlayer to compact controls when the box is too tight for full transport + volume + fullscreen. */
const COMPACT_VIDEO_MIN_WIDTH = 280;
const COMPACT_VIDEO_MIN_HEIGHT = 220;

const LazyCurrentImageVideoSurface = lazy(() => import('./CurrentImageVideoSurface'));

interface DisplayedGalleryMedia {
  item: GalleryItem;
  /** Full selected GalleryItem opens the menu without refetching; legacy recentImages fall back to a ref. */
  menuTarget: GalleryItem | GalleryItemRef;
}

/**
 * Pick the displayed live frame via the queue feature's shared auto-switch policy (`getAutoSwitchFollowedLive`), while
 * tracking percentage only on an actively running session so a settling slot never shows stale progress under
 * "Generating" while a sibling slot is starting.
 */
const resolveCurrentImageActiveSession = (
  sessions: readonly QueueActiveSession[]
): { liveSession: QueueActiveSession | null; trackedSession: QueueActiveSession | null } => {
  const liveSession = getAutoSwitchFollowedLive('progress', sessions, null);
  const trackedSession =
    (liveSession?.state === 'running' ? liveSession : null) ??
    sessions.filter((session) => session.state === 'running').at(-1) ??
    null;
  return { liveSession, trackedSession };
};

const resolveProgressPercentage = ({
  activeProgress,
  activeTarget,
  slotProgressPercentage,
}: {
  activeProgress: Pick<QueueItemProgress, 'activeItemIndex' | 'percentage' | 'slotPercentages'> | null | undefined;
  activeTarget: { itemIndex: number; queueItemId: string } | null;
  slotProgressPercentage?: number | null;
}): number | null => {
  if (activeTarget && activeProgress) {
    if (activeProgress.slotPercentages && Object.hasOwn(activeProgress.slotPercentages, activeTarget.itemIndex)) {
      return activeProgress.slotPercentages[activeTarget.itemIndex] ?? null;
    }
    if (activeProgress.activeItemIndex === activeTarget.itemIndex) {
      return activeProgress.percentage;
    }
  }
  return slotProgressPercentage ?? null;
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
  const flowStore = useStoreApi();
  const startDrag = usePointerDrag();

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
  const hasLiveFrame = Boolean(projectProgressImage);
  const isGenerating = hasLiveFrame || hasRunningSession;
  const isWaiting = !isGenerating && isWorkflowRunning;
  const isShowingVideo = !hasLiveFrame && displayedItem?.kind === 'video';
  const progressPercent =
    isGenerating && typeof progressPercentage === 'number' && progressPercentage > 0
      ? Math.round(Math.min(100, Math.max(0, progressPercentage * 100)))
      : null;

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
  const isCompactHeader = displayWidth < COMPACT_HEADER_WIDTH;
  const videoPlayerSize =
    displayWidth < COMPACT_VIDEO_MIN_WIDTH || displayHeight < COMPACT_VIDEO_MIN_HEIGHT ? 'compact' : 'full';

  const videoControllerRef = useRef<CurrentImageVideoController | null>(null);
  const resizeHandleRef = useRef<HTMLDivElement | null>(null);
  const activeDragRef = useRef<AbortController | null>(null);
  const [copyReadyVideoKey, setCopyReadyVideoKey] = useState<string | null>(null);
  const canCopyVideoFrame = isShowingVideo && videoSurfaceKey !== null && copyReadyVideoKey === videoSurfaceKey;

  // Cancel any in-flight resize drag on unmount or if the node or active project changes underneath it.
  useEffect(() => {
    activeDragRef.current?.abort();
    activeDragRef.current = null;
    return () => {
      activeDragRef.current?.abort();
      activeDragRef.current = null;
    };
  }, [node.id, projectId]);

  const handleVideoControllerChange = useCallback((controller: CurrentImageVideoController | null) => {
    videoControllerRef.current = controller;
  }, []);

  const handleVideoCopyAvailabilityChange = useCallback((surfaceKey: string, isAvailable: boolean) => {
    setCopyReadyVideoKey((prev) => (isAvailable ? surfaceKey : prev === surfaceKey ? null : prev));
  }, []);

  const handleCopyCurrentFrame = useCallback((event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    void videoControllerRef.current?.copyCurrentFrame();
  }, []);

  const resetSize = useCallback(
    (event: MouseEvent<HTMLElement> | KeyboardEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      activeDragRef.current?.abort();
      activeDragRef.current = null;
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
      event.stopPropagation();
      activeDragRef.current?.abort();

      const handle = event.currentTarget;
      const zoom = flowStore.getState().transform[2] || 1;
      const startWidth = savedWidth;
      const startHeight = savedHeight;
      let hasCrossedThreshold = false;
      let liveWidth = startWidth;
      let liveHeight = startHeight;

      const session = startDrag(event, {
        cursor: 'nwse-resize',
        onEnd: (reason) => {
          if (activeDragRef.current !== session) {
            return;
          }
          activeDragRef.current = null;
          setLiveSizeState(null);
          if (reason !== 'release' || !hasCrossedThreshold) {
            return;
          }
          if (liveWidth !== startWidth || liveHeight !== startHeight) {
            editGraph({ height: liveHeight, nodeId: node.id, type: 'setCurrentImageNodeSize', width: liveWidth });
          }
        },
        onMove: (deltaX, deltaY) => {
          if (activeDragRef.current !== session) {
            return;
          }
          if (!hasCrossedThreshold && Math.hypot(deltaX, deltaY) < RESIZE_DRAG_THRESHOLD_PX) {
            return;
          }
          hasCrossedThreshold = true;
          liveWidth = clampCurrentImageWidth(startWidth + deltaX / zoom);
          liveHeight = clampCurrentImageHeight(startHeight + deltaY / zoom);
          setLiveSizeState({ height: liveHeight, nodeId: node.id, projectId, width: liveWidth });
        },
      });

      activeDragRef.current = session;
      handle.setAttribute('data-dragging', '');
      trackResizeDrag(session.signal);
      session.signal.addEventListener('abort', () => {
        handle.removeAttribute('data-dragging');
        if (activeDragRef.current === session) {
          activeDragRef.current = null;
          setLiveSizeState(null);
        }
      });
    },
    [editGraph, flowStore, node.id, projectId, savedHeight, savedWidth, startDrag]
  );

  const handleResizeKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const step = event.shiftKey ? KEYBOARD_RESIZE_STEP_PX * 2 : KEYBOARD_RESIZE_STEP_PX;
      const targets: Partial<Record<string, [number, number]>> = {
        ArrowDown: [savedWidth, savedHeight + step],
        ArrowLeft: [savedWidth - step, savedHeight],
        ArrowRight: [savedWidth + step, savedHeight],
        ArrowUp: [savedWidth, savedHeight - step],
        End: [CURRENT_IMAGE_MAX_SIZE, CURRENT_IMAGE_MAX_SIZE],
        Home: [CURRENT_IMAGE_MIN_WIDTH, CURRENT_IMAGE_MIN_HEIGHT],
      };
      const target = targets[event.key];
      if (!target) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      const nextWidth = clampCurrentImageWidth(target[0]);
      const nextHeight = clampCurrentImageHeight(target[1]);
      if (nextWidth !== savedWidth || nextHeight !== savedHeight) {
        editGraph({ height: nextHeight, nodeId: node.id, type: 'setCurrentImageNodeSize', width: nextWidth });
      }
    },
    [editGraph, node.id, savedHeight, savedWidth]
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
  const videoPosterFallback = useMemo(() => {
    if (!displayedItem || displayedItem.kind !== 'video') {
      return null;
    }
    const posterSource = imageUrlToStreamingSource({
      alt: displayedItem.name,
      height: displayedItem.height,
      kind: 'fallback',
      src: displayedItem.thumbnailUrl,
      width: displayedItem.width,
    });
    return (
      <Box bg="black" className="nodrag nopan nowheel" h="full" position="relative" w="full">
        <StreamingImageFrame fallbackImage={posterSource} fit="contain" h="full" liveImage={null} w="full" />
      </Box>
    );
  }, [displayedItem]);
  const liveImage = useMemo(() => progressImageToStreamingSource(projectProgressImage), [projectProgressImage]);

  const copyUnavailableReason = !canCopyImage()
    ? t('common.copyUnavailable')
    : !canCopyVideoFrame
      ? t('widgets.preview.copyCurrentFrameNotReady')
      : undefined;
  const resetSizeLabel = t('widgets.workflow.currentImageResetSize', {
    height: displayHeight,
    width: displayWidth,
  });
  const showItemMenuButton = Boolean(openItemMenu);
  const endsWithHeaderButton = showItemMenuButton || (isCustomSize ? isCompactHeader : isShowingVideo);

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
      <Flex {...getWorkflowNodeHeaderProps({ endsWithButton: endsWithHeaderButton })} flexShrink={0} gap="1.5">
        <MiddleTruncate data-testid="current-image-node-title" flex="1" fontWeight="600" minW="0" text={title} />
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
            <WorkflowNodeHeaderButton
              aria-label={t('widgets.preview.copyCurrentFrame')}
              color="fg.muted"
              content={t('widgets.preview.copyCurrentFrame')}
              data-testid="current-image-copy-video-frame"
              unavailableReason={copyUnavailableReason}
              onClick={handleCopyCurrentFrame}
            >
              <Icon as={CopyIcon} boxSize="3.5" />
            </WorkflowNodeHeaderButton>
          ) : null}
          {isCustomSize ? (
            isCompactHeader ? (
              <WorkflowNodeHeaderButton
                aria-label={resetSizeLabel}
                color="fg.muted"
                content={resetSizeLabel}
                data-testid="current-image-node-reset-size"
                onClick={resetSize}
              >
                <Icon as={RotateCcwIcon} boxSize="3.5" />
              </WorkflowNodeHeaderButton>
            ) : (
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
            )
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
          <Suspense fallback={videoPosterFallback}>
            <LazyCurrentImageVideoSurface
              key={videoSurfaceKey}
              item={displayedItem}
              nodeId={node.id}
              playerSize={videoPlayerSize}
              surfaceKey={videoSurfaceKey}
              videoDragRef={videoDragRef}
              onControllerChange={handleVideoControllerChange}
              onCopyAvailabilityChange={handleVideoCopyAvailabilityChange}
            />
          </Suspense>
        ) : (
          <DraggableCurrentImage itemRef={draggableImageRef}>
            <StreamingImageFrame fallbackImage={fallbackImage} fit="contain" h="full" liveImage={liveImage} w="full">
              <Flex align="center" color="fg.subtle" fontSize="xs" h="full" justify="center" px="4" textAlign="center">
                {t('widgets.workflow.currentImageEmpty')}
              </Flex>
            </StreamingImageFrame>
          </DraggableCurrentImage>
        )}

        {isGenerating ? (
          <Flex
            align="center"
            aria-hidden="true"
            bg="bg/85"
            bottom={isShowingVideo ? undefined : '2'}
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
            top={isShowingVideo ? '2' : undefined}
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

      {/* Dedicated footer strip keeps the platform ResizeCorner clear of VideoPlayer's Fullscreen control. */}
      <Flex
        align="center"
        bg="bg.subtle"
        borderTopWidth="1px"
        className="nodrag nopan"
        flexShrink={0}
        h="4"
        position="relative"
        w="full"
      >
        <ResizeCorner
          ref={resizeHandleRef}
          className="nodrag nopan"
          data-testid="current-image-node-resize-handle"
          label={t('widgets.workflow.currentImageResizeHandle')}
          valueMax={CURRENT_IMAGE_MAX_SIZE}
          valueMin={CURRENT_IMAGE_MIN_WIDTH}
          valueNow={displayWidth}
          valueText={t('widgets.floating.resizeValue', { height: displayHeight, width: displayWidth })}
          onDoubleClick={resetSize}
          onKeyDown={handleResizeKeyDown}
          onPointerDown={handleResizePointerDown}
        />
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

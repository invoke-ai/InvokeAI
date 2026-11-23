import type {
  GalleryItem,
  GalleryItemRef,
  GalleryVideoItem,
  GeneratedImageContract,
} from '@features/gallery/contracts';
import type { LatestProgressImageSnapshot } from '@features/queue/react';
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
import {
  useActiveProgressTargets,
  useFollowedProgressTargets,
  useFramedProgressTargets,
  useItemProgress,
  useProgressImage,
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
import {
  useWorkflowNotifications,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from '@features/workflow/ui/WorkflowUiContext';
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
  useSyncExternalStore,
  type CSSProperties,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
} from 'react';
import { useTranslation } from 'react-i18next';

import type { CurrentImageFlowNode as CurrentImageFlowNodeType } from './flowAdapters';

const KEYBOARD_RESIZE_STEP_PX = 32;
// Seek far enough from zero to clear poster state while staying within frame zero.
const FIRST_FRAME_SEEK_SECONDS = 0.0001;

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

const LazyVideoPlayer = lazy(() =>
  import('@platform/ui/video-player/VideoPlayer').then((mod) => ({ default: mod.VideoPlayer }))
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

const isSameProgressTarget = (
  left: { itemIndex: number; queueItemId: string },
  right: { itemIndex: number; queueItemId: string }
): boolean => left.queueItemId === right.queueItemId && left.itemIndex === right.itemIndex;

// Match the displayed frame's framed running/settling target, keep the previously displayed local slot across foreign updates
// (preferring running framed slots over settling ones), or fall back to a running/settling slot in the displayed project.
export const resolveProgressTarget = (
  progressImage: LatestProgressImageSnapshot | null | undefined,
  followedTargets: readonly { itemIndex: number; queueItemId: string }[],
  projectQueueItemIds?: ReadonlySet<string> | null,
  framedTargets?: readonly { itemIndex: number; queueItemId: string }[],
  preferredTarget?: { itemIndex: number; queueItemId: string } | null,
  runningTargets?: readonly { itemIndex: number; queueItemId: string }[]
): { itemIndex: number; queueItemId: string } | null => {
  const matchesProject = (queueItemId: string): boolean => !projectQueueItemIds || projectQueueItemIds.has(queueItemId);
  const isFollowedInProject = (target: { itemIndex: number; queueItemId: string }): boolean =>
    matchesProject(target.queueItemId) && followedTargets.some((candidate) => isSameProgressTarget(candidate, target));
  const isFramedInProject = (target: { itemIndex: number; queueItemId: string }): boolean =>
    isFollowedInProject(target) &&
    (!framedTargets || framedTargets.some((candidate) => isSameProgressTarget(candidate, target)));
  const isRunning = (target: { itemIndex: number; queueItemId: string }): boolean =>
    !runningTargets || runningTargets.some((candidate) => isSameProgressTarget(candidate, target));

  const frameTarget = progressImage?.target;
  if (frameTarget && isFramedInProject(frameTarget)) {
    return frameTarget;
  }

  const framedInProject = framedTargets ? framedTargets.filter(isFollowedInProject) : [];
  const runningFramedInProject = framedInProject.filter(isRunning);

  if (preferredTarget && isFramedInProject(preferredTarget)) {
    // Do not let a finished/settling preferred slot block a live running framed slot in the same project.
    if (isRunning(preferredTarget) || runningFramedInProject.length === 0) {
      return preferredTarget;
    }
  }

  const latestRunningFramed = runningFramedInProject.at(-1);
  if (latestRunningFramed) {
    return latestRunningFramed;
  }
  const latestFramed = framedInProject.at(-1);
  if (latestFramed) {
    return latestFramed;
  }
  if (preferredTarget && isFollowedInProject(preferredTarget)) {
    return preferredTarget;
  }
  return followedTargets.find((target) => matchesProject(target.queueItemId)) ?? null;
};

// Keep this project's retained frame only while the slot is framed in the current attempt, and drop frames on requeue/settlement clear.
export const resolveLiveProgressImage = ({
  activeTarget,
  allowUntargetedFrame = false,
  followedTargets,
  framedTargets,
  progressImage,
  slotProgressImage,
}: {
  activeTarget: { itemIndex: number; queueItemId: string } | null;
  allowUntargetedFrame?: boolean;
  followedTargets: readonly { itemIndex: number; queueItemId: string }[];
  framedTargets: readonly { itemIndex: number; queueItemId: string }[];
  progressImage: LatestProgressImageSnapshot | null | undefined;
  slotProgressImage: Omit<LatestProgressImageSnapshot, 'target'> | null | undefined;
}): Omit<LatestProgressImageSnapshot, 'target'> | null => {
  if (!activeTarget) {
    return allowUntargetedFrame && progressImage && !progressImage.target ? progressImage : null;
  }
  const isActiveTargetFramed = framedTargets.some((candidate) => isSameProgressTarget(candidate, activeTarget));
  if (slotProgressImage && isActiveTargetFramed) {
    return slotProgressImage;
  }
  if (!progressImage) {
    return null;
  }
  if (!progressImage.target) {
    return allowUntargetedFrame && followedTargets.length > 0 ? progressImage : null;
  }
  return isActiveTargetFramed && isSameProgressTarget(progressImage.target, activeTarget) ? progressImage : null;
};

// Resolve the 1-based batch slot's backend item ID for per-slot progress lookup.
export const resolveActiveBackendItemId = (
  activeTarget: { itemIndex: number; queueItemId: string } | null,
  projectQueueItemsById: ReadonlyMap<string, { backendItemIds?: readonly number[]; id: string }> | null
): number | null => {
  if (!activeTarget || activeTarget.itemIndex <= 0 || !projectQueueItemsById) {
    return null;
  }
  return projectQueueItemsById.get(activeTarget.queueItemId)?.backendItemIds?.[activeTarget.itemIndex - 1] ?? null;
};

// Prefer the coordinator's percentage when its active slot matches the displayed frame; use per-slot lookup otherwise.
export const resolveProgressPercentage = ({
  activeProgress,
  activeTarget,
  slotProgressPercentage,
}: {
  activeProgress: { activeItemIndex?: number; percentage: number | null } | null | undefined;
  activeTarget: { itemIndex: number; queueItemId: string } | null;
  slotProgressPercentage?: number | null;
}): number | null => {
  if (!activeProgress) {
    return slotProgressPercentage ?? null;
  }

  const hasTargetSlotIndex = activeTarget !== null && activeTarget.itemIndex > 0;
  if (hasTargetSlotIndex) {
    if (activeProgress.activeItemIndex === activeTarget.itemIndex) {
      return activeProgress.percentage;
    }
    return slotProgressPercentage ?? null;
  }

  return activeProgress.percentage ?? slotProgressPercentage ?? null;
};

export interface CurrentImageExecutionState {
  dragRef: GalleryItemRef | null;
  hasLiveFrame: boolean;
  isGenerating: boolean;
  isShowingVideo: boolean;
  isWaiting: boolean;
  progressPercent: number | null;
}

// Keep queued workflow runs in Waiting until a job owned by this project is actively executing.
export const resolveCurrentImageExecutionState = ({
  activeTargetCount,
  displayedItem,
  isWorkflowRunning,
  progressImage,
  progressPercentage,
}: {
  activeTargetCount: number;
  displayedItem: GalleryItem | null;
  isWorkflowRunning: boolean;
  progressImage: unknown;
  progressPercentage: number | null | undefined;
}): CurrentImageExecutionState => {
  const hasLiveFrame = Boolean(progressImage);
  const isGenerating = hasLiveFrame || activeTargetCount > 0;
  const isWaiting = !isGenerating && isWorkflowRunning;
  const isShowingVideo = !hasLiveFrame && displayedItem?.kind === 'video';
  const dragRef = !hasLiveFrame && displayedItem ? toGalleryItemRef(displayedItem) : null;
  const progressPercent =
    isGenerating && typeof progressPercentage === 'number' && progressPercentage > 0
      ? Math.round(Math.min(100, Math.max(0, progressPercentage * 100)))
      : null;

  return { dragRef, hasLiveFrame, isGenerating, isShowingVideo, isWaiting, progressPercent };
};

export interface VideoErrorRecovery {
  handleError(video: HTMLVideoElement | null, isElementCurrent: () => boolean): void;
  retry(video: HTMLVideoElement | null): void;
}

// Refresh the protected-media cookie once per media selection (until manual retry) before surfacing manual retry.
export const createVideoErrorRecovery = ({
  getAccountEpoch,
  onFailedChange,
  refreshCookie,
}: {
  getAccountEpoch: () => number;
  onFailedChange: (failed: boolean) => void;
  refreshCookie: () => Promise<boolean>;
}): VideoErrorRecovery => {
  let automaticRefreshUsed = false;
  let pendingRefresh: Promise<boolean> | null = null;

  return {
    handleError(video, isElementCurrent) {
      if (!video || pendingRefresh !== null) {
        return;
      }

      if (automaticRefreshUsed) {
        onFailedChange(true);
        return;
      }

      automaticRefreshUsed = true;
      const accountEpoch = getAccountEpoch();
      const refresh = refreshCookie();
      pendingRefresh = refresh;

      const finishRefresh = (refreshed: boolean): void => {
        if (pendingRefresh !== refresh) {
          return;
        }

        pendingRefresh = null;

        if (!isElementCurrent() || getAccountEpoch() !== accountEpoch) {
          return;
        }

        if (refreshed) {
          video.load();
        } else {
          onFailedChange(true);
        }
      };

      void refresh.then(finishRefresh, () => finishRefresh(false));
    },
    retry(video) {
      if (!video) {
        return;
      }

      automaticRefreshUsed = false;
      pendingRefresh = null;
      onFailedChange(false);
      video.load();
    },
  };
};

const resolveDisplayedGalleryItem = (values: Record<string, unknown>): GalleryItem | null => {
  const selected = getSelectedGalleryItemFromValues(values);
  if (selected) {
    return selected;
  }

  const recentImages = Array.isArray(values.recentImages) ? (values.recentImages as GeneratedImageContract[]) : [];
  const latest = recentImages[0];

  return latest ? legacyGeneratedImageToGalleryItem(latest) : null;
};

const isSameDisplayedGalleryItem = (a: GalleryItem | null, b: GalleryItem | null): boolean =>
  a === b ||
  (a !== null &&
    b !== null &&
    a.kind === b.kind &&
    a.name === b.name &&
    a.fullUrl === b.fullUrl &&
    a.thumbnailUrl === b.thumbnailUrl &&
    a.width === b.width &&
    a.height === b.height &&
    (a.kind !== 'video' || b.kind !== 'video' || a.fps === b.fps));

const isVideoFrameReady = (video: HTMLVideoElement | null): video is HTMLVideoElement =>
  Boolean(
    video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0 && video.videoHeight > 0
  );

const isCanvasSecurityError = (error: unknown): boolean =>
  error instanceof DOMException && error.name === 'SecurityError';

const encodeCanvasPng = (canvas: HTMLCanvasElement): Promise<Blob | null> =>
  new Promise((resolve, reject) => {
    try {
      canvas.toBlob(resolve, 'image/png');
    } catch (error: unknown) {
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });

const seekVideo = (video: HTMLVideoElement, seconds: number): void => {
  try {
    video.currentTime = seconds;
  } catch {
    // Some browsers throw while the seekable range is still empty; the element keeps its position.
  }
};

/** The image under the node's frame, draggable wherever gallery images can be dropped. */
const DraggableCurrentImage = ({
  children,
  itemRef,
  onContextMenu,
}: {
  children: ReactNode;
  itemRef: GalleryItemRef | null;
  onContextMenu: ((event: MouseEvent<HTMLElement>) => void) | undefined;
}) => {
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
    <Box
      ref={setNodeRef}
      className="nodrag"
      cursor={itemRef ? 'grab' : undefined}
      h="full"
      w="full"
      onContextMenu={onContextMenu}
      {...listeners}
    >
      {children}
    </Box>
  );
};

const CurrentImageEditorNode = ({ data, selected }: NodeProps<CurrentImageFlowNodeType>) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const notifications = useWorkflowNotifications();
  const { nodeExecution } = useWorkflowUi();
  // Select only the shown media item so unrelated gallery changes (board filter, page, scroll) do not re-render.
  const displayedItem = useWorkflowProjectSelector(
    (project) => resolveDisplayedGalleryItem(project.galleryValues),
    isSameDisplayedGalleryItem
  );
  const projectId = useWorkflowProjectSelector((project) => project.id);
  const isWorkflowRunning = useWorkflowProjectSelector((project) => project.isWorkflowRunning);
  const queueItems = useWorkflowProjectSelector((project) => project.queueItems);
  const getOriginProjectId = useCallback(() => nodeExecution.getOrigin()?.projectId ?? null, [nodeExecution]);
  const originProjectId = useSyncExternalStore(nodeExecution.subscribeOrigin, getOriginProjectId, getOriginProjectId);

  const progressImage = useProgressImage();
  const activeTargets = useActiveProgressTargets();
  const followedTargets = useFollowedProgressTargets();
  const framedTargets = useFramedProgressTargets();

  const isOriginInProject = originProjectId === null || originProjectId === projectId;
  const projectQueueItemsById = useMemo(
    () => (queueItems ? new Map(queueItems.map((item) => [item.id, item])) : null),
    [queueItems]
  );
  const projectQueueItemIds = useMemo(
    () => (projectQueueItemsById ? new Set(projectQueueItemsById.keys()) : null),
    [projectQueueItemsById]
  );

  // When project queue items are known, match by item ID directly; use nodeExecution origin only as a fallback.
  const isTargetInProject = useCallback(
    (queueItemId: string): boolean => (projectQueueItemIds ? projectQueueItemIds.has(queueItemId) : isOriginInProject),
    [isOriginInProject, projectQueueItemIds]
  );

  const projectActiveTargets = useMemo(
    () => activeTargets.filter((target) => isTargetInProject(target.queueItemId)),
    [activeTargets, isTargetInProject]
  );

  const projectFollowedTargets = useMemo(
    () => followedTargets.filter((target) => isTargetInProject(target.queueItemId)),
    [followedTargets, isTargetInProject]
  );

  const projectFramedTargets = useMemo(
    () => framedTargets.filter((target) => isTargetInProject(target.queueItemId)),
    [framedTargets, isTargetInProject]
  );

  const [lastDisplayedTarget, setLastDisplayedTarget] = useState<{
    projectId: string;
    target: { itemIndex: number; queueItemId: string } | null;
  }>({ projectId, target: null });
  const preferredTarget = lastDisplayedTarget.projectId === projectId ? lastDisplayedTarget.target : null;

  const activeTarget = useMemo(
    () =>
      resolveProgressTarget(
        progressImage,
        projectFollowedTargets,
        null,
        projectFramedTargets,
        preferredTarget,
        projectActiveTargets
      ),
    [preferredTarget, progressImage, projectActiveTargets, projectFollowedTargets, projectFramedTargets]
  );

  const isSameDisplayedTarget =
    lastDisplayedTarget.projectId === projectId &&
    ((lastDisplayedTarget.target === null && activeTarget === null) ||
      (lastDisplayedTarget.target !== null &&
        activeTarget !== null &&
        isSameProgressTarget(lastDisplayedTarget.target, activeTarget)));
  if (!isSameDisplayedTarget) {
    setLastDisplayedTarget({ projectId, target: activeTarget });
  }

  const slotProgressImage = useQueueItemProgressImage(activeTarget?.queueItemId ?? '', activeTarget?.itemIndex ?? 0);
  const projectProgressImage = useMemo(
    () =>
      resolveLiveProgressImage({
        activeTarget,
        allowUntargetedFrame: projectQueueItemIds === null && isOriginInProject,
        followedTargets: projectFollowedTargets,
        framedTargets: projectFramedTargets,
        progressImage,
        slotProgressImage,
      }),
    [
      activeTarget,
      isOriginInProject,
      progressImage,
      projectFollowedTargets,
      projectFramedTargets,
      projectQueueItemIds,
      slotProgressImage,
    ]
  );
  const activeBackendItemId = useMemo(
    () => resolveActiveBackendItemId(activeTarget, projectQueueItemsById),
    [activeTarget, projectQueueItemsById]
  );

  const activeProgress = useQueueItemProgress(activeTarget?.queueItemId ?? '');
  const slotProgress = useItemProgress(activeBackendItemId);
  const progressPercentage = useMemo(
    () =>
      resolveProgressPercentage({
        activeProgress,
        activeTarget,
        slotProgressPercentage: slotProgress?.percentage,
      }),
    [activeProgress, activeTarget, slotProgress?.percentage]
  );

  const displayedItemKey = useMemo(() => (displayedItem ? toGalleryItemKey(displayedItem) : null), [displayedItem]);
  const { hasLiveFrame, isGenerating, isShowingVideo, isWaiting, progressPercent } = useMemo(
    () =>
      resolveCurrentImageExecutionState({
        activeTargetCount: projectActiveTargets.length,
        displayedItem,
        isWorkflowRunning,
        progressImage: projectProgressImage,
        progressPercentage,
      }),
    [displayedItem, isWorkflowRunning, progressPercentage, projectActiveTargets.length, projectProgressImage]
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
      if (!canCopyImage()) {
        notifications.error(t('widgets.preview.copyCurrentFrameUnsupported'));
        return;
      }
      if (!isVideoFrameReady(video)) {
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
        blob = await encodeCanvasPng(canvas);
      } catch (error: unknown) {
        notifications.error(
          t(
            isCanvasSecurityError(error)
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
  const openItemMenu = useGalleryItemContextMenu(shownMediaRef);
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
            <Text color="brand.solid" data-node-status-indicator="true" fontSize="xs" fontVariantNumeric="tabular-nums">
              {progressPercent !== null
                ? `${t('widgets.workflow.generating')} ${progressPercent}%`
                : t('widgets.workflow.generating')}
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

      <Box flex="1" minH="0" overflow="hidden" position="relative" w="full">
        {isShowingVideo && displayedItem?.kind === 'video' && videoSurfaceKey ? (
          <CurrentImageVideoSurface
            key={videoSurfaceKey}
            item={displayedItem}
            surfaceKey={videoSurfaceKey}
            onContextMenu={openItemMenu}
            onCopyAvailabilityChange={handleVideoCopyAvailabilityChange}
            onVideoElementChange={handleVideoElementChange}
          />
        ) : (
          <DraggableCurrentImage itemRef={draggableImageRef} onContextMenu={openItemMenu}>
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

const CurrentImageVideoSurface = ({
  item,
  onContextMenu,
  onCopyAvailabilityChange,
  onVideoElementChange,
  surfaceKey,
}: {
  item: GalleryVideoItem;
  onContextMenu?: (event: MouseEvent<HTMLElement>) => void;
  onCopyAvailabilityChange: (surfaceKey: string, isAvailable: boolean) => void;
  onVideoElementChange: (element: HTMLVideoElement | null) => void;
  surfaceKey: string;
}) => {
  const { t } = useTranslation();
  const localVideoRef = useRef<HTMLVideoElement | null>(null);
  const [hasFailed, setHasFailed] = useState(false);

  const publishCopyAvailability = useCallback(() => {
    onCopyAvailabilityChange(surfaceKey, canCopyImage() && isVideoFrameReady(localVideoRef.current));
  }, [onCopyAvailabilityChange, surfaceKey]);

  const setVideoElementRef = useCallback(
    (element: HTMLVideoElement | null) => {
      localVideoRef.current = element;
      onVideoElementChange(element);
      if (element) {
        element.disablePictureInPicture = true;
      }
      onCopyAvailabilityChange(surfaceKey, canCopyImage() && isVideoFrameReady(element));
    },
    [onCopyAvailabilityChange, onVideoElementChange, surfaceKey]
  );

  const recovery = useMemo(
    () =>
      createVideoErrorRecovery({
        getAccountEpoch: () => getAuthSession().accountEpoch,
        onFailedChange: setHasFailed,
        refreshCookie: refreshProtectedMediaCookie,
      }),
    []
  );

  const handleVideoError = useCallback(() => {
    publishCopyAvailability();
    const video = localVideoRef.current;
    recovery.handleError(video, () => localVideoRef.current === video);
  }, [publishCopyAvailability, recovery]);

  const handleLoadedData = useCallback(() => {
    publishCopyAvailability();
  }, [publishCopyAvailability]);

  const handleLoadedMetadata = useCallback(() => {
    publishCopyAvailability();
    const video = localVideoRef.current;
    if (video && video.paused && video.currentTime === 0) {
      seekVideo(video, FIRST_FRAME_SEEK_SECONDS);
    }
  }, [publishCopyAvailability]);

  const handleRetryClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
      recovery.retry(localVideoRef.current);
      publishCopyAvailability();
    },
    [publishCopyAvailability, recovery]
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
    <Box
      bg="black"
      className="nodrag nopan nowheel"
      h="full"
      position="relative"
      w="full"
      onContextMenu={onContextMenu}
    >
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
          onLoadedData={handleLoadedData}
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

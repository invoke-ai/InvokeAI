import type { GalleryItemRef, GalleryVideoItem } from '@features/gallery/contracts';

import { Badge, Box } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { getAuthSession, refreshProtectedMediaCookie } from '@features/identity';
import { useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import { Tooltip } from '@platform/ui/Tooltip';
import { useVideoErrorRecovery, VideoFailureOverlay } from '@platform/ui/video-player/VideoFailureOverlay';
import { VIDEO_FRAME_COPY_FAILURE_KEYS } from '@platform/ui/video-player/videoFrameCopyKeys';
import {
  copyVideoFrame,
  FIRST_FRAME_SEEK_SECONDS,
  isVideoFrameCopyAvailable,
  seekVideo,
} from '@platform/ui/video-player/videoFrameUtils';
import { VideoPlayer } from '@platform/ui/video-player/VideoPlayer';
import { GripHorizontalIcon } from 'lucide-react';
import { memo, useCallback, useImperativeHandle, useMemo, useRef, type MouseEvent, type Ref } from 'react';
import { useTranslation } from 'react-i18next';

const getAccountEpoch = (): number => getAuthSession().accountEpoch;

export interface CurrentImageVideoController {
  copyCurrentFrame(): Promise<void>;
}

export interface CurrentImageVideoSurfaceProps {
  controllerRef: Ref<CurrentImageVideoController>;
  item: GalleryVideoItem;
  nodeId: string;
  onCopyAvailabilityChange: (surfaceKey: string, isAvailable: boolean) => void;
  playerSize: 'compact' | 'full';
  surfaceKey: string;
  videoDragRef: GalleryItemRef | null;
}

const CurrentImageVideoSurfaceComponent = ({
  controllerRef,
  item,
  nodeId,
  onCopyAvailabilityChange,
  playerSize,
  surfaceKey,
  videoDragRef,
}: CurrentImageVideoSurfaceProps) => {
  const { t } = useTranslation();
  const notifications = useWorkflowNotifications();
  const localVideoRef = useRef<HTMLVideoElement | null>(null);
  const { handleError, hasFailed, retry } = useVideoErrorRecovery({
    getAccountEpoch,
    refreshCookie: refreshProtectedMediaCookie,
  });

  const publishCopyAvailability = useCallback(() => {
    onCopyAvailabilityChange(surfaceKey, isVideoFrameCopyAvailable(localVideoRef.current));
  }, [onCopyAvailabilityChange, surfaceKey]);

  const setVideoElementRef = useCallback(
    (element: HTMLVideoElement | null) => {
      localVideoRef.current = element;
      if (element) {
        element.disablePictureInPicture = true;
      }
      onCopyAvailabilityChange(surfaceKey, isVideoFrameCopyAvailable(element));
    },
    [onCopyAvailabilityChange, surfaceKey]
  );

  const copyCurrentFrame = useCallback(async () => {
    const video = localVideoRef.current;
    const accountEpoch = getAccountEpoch();
    const result = await copyVideoFrame(
      video,
      () => localVideoRef.current === video && getAccountEpoch() === accountEpoch
    );

    if (result.ok) {
      notifications.success(t('widgets.preview.copyCurrentFrameSuccess'));
      return;
    }

    notifications.error(t(VIDEO_FRAME_COPY_FAILURE_KEYS[result.reason]));
  }, [notifications, t]);

  useImperativeHandle(controllerRef, () => ({ copyCurrentFrame }), [copyCurrentFrame]);

  const handleVideoError = useCallback(() => {
    publishCopyAvailability();
    const video = localVideoRef.current;
    handleError(video, () => localVideoRef.current === video);
  }, [handleError, publishCopyAvailability]);

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
      retry(localVideoRef.current);
      publishCopyAvailability();
    },
    [publishCopyAvailability, retry]
  );

  const failureOverlay = useMemo(
    () =>
      hasFailed ? (
        <VideoFailureOverlay
          height={item.height}
          poster={item.thumbnailUrl}
          size="compact"
          testId="current-image-video-failure"
          width={item.width}
          onRetry={handleRetryClick}
        />
      ) : null,
    [handleRetryClick, hasFailed, item.height, item.thumbnailUrl, item.width]
  );

  return (
    <Box bg="black" className="nodrag nopan nowheel" h="full" position="relative" w="full">
      <VideoPlayer
        ref={setVideoElementRef}
        aria-label={item.name}
        defaultLoop
        fps={item.fps}
        overlay={failureOverlay}
        poster={item.thumbnailUrl}
        size={playerSize}
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
      {videoDragRef ? <CurrentMediaDragGrip itemRef={videoDragRef} nodeId={nodeId} /> : null}
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

export const CurrentImageVideoSurface = memo(CurrentImageVideoSurfaceComponent);
export default CurrentImageVideoSurface;

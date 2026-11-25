import type { GalleryItemRef, GalleryVideoItem } from '@features/gallery/contracts';

import { Badge, Box, Flex, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { getAuthSession, refreshProtectedMediaCookie } from '@features/identity';
import { useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import { Button } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  copyVideoFrame,
  createVideoErrorRecovery,
  FIRST_FRAME_SEEK_SECONDS,
  isVideoFrameCopyAvailable,
  seekVideo,
  type VideoErrorRecovery,
  type VideoFrameCopyFailureReason,
} from '@platform/ui/video-player/videoFrameUtils';
import { VideoPlayer } from '@platform/ui/video-player/VideoPlayer';
import { GripHorizontalIcon } from 'lucide-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

const VIDEO_FAILURE_POSTER_STYLE: CSSProperties = {
  display: 'block',
  height: '100%',
  inset: 0,
  objectFit: 'contain',
  position: 'absolute',
  width: '100%',
};

const VIDEO_FRAME_COPY_FAILURE_KEYS: Record<VideoFrameCopyFailureReason, string> = {
  'clipboard-failed': 'widgets.preview.copyCurrentFrameWriteFailed',
  'draw-failed': 'widgets.preview.copyCurrentFrameDrawFailed',
  'encode-failed': 'widgets.preview.copyCurrentFrameEncodeFailed',
  'not-ready': 'widgets.preview.copyCurrentFrameNotReady',
  stale: 'widgets.preview.copyCurrentFrameStale',
  unsupported: 'widgets.preview.copyCurrentFrameUnsupported',
};

export interface CurrentImageVideoController {
  copyCurrentFrame(): Promise<void>;
}

export interface CurrentImageVideoSurfaceProps {
  item: GalleryVideoItem;
  nodeId: string;
  onControllerChange: (controller: CurrentImageVideoController | null) => void;
  onCopyAvailabilityChange: (surfaceKey: string, isAvailable: boolean) => void;
  playerSize: 'compact' | 'full';
  surfaceKey: string;
  videoDragRef: GalleryItemRef | null;
}

const CurrentImageVideoSurfaceComponent = ({
  item,
  nodeId,
  onControllerChange,
  onCopyAvailabilityChange,
  playerSize,
  surfaceKey,
  videoDragRef,
}: CurrentImageVideoSurfaceProps) => {
  const { t } = useTranslation();
  const notifications = useWorkflowNotifications();
  const localVideoRef = useRef<HTMLVideoElement | null>(null);
  const [hasFailed, setHasFailed] = useState(false);

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

  const recoveryRef = useRef<VideoErrorRecovery | null>(null);
  const getRecovery = useCallback(() => {
    if (!recoveryRef.current) {
      recoveryRef.current = createVideoErrorRecovery({
        getAccountEpoch: () => getAuthSession().accountEpoch,
        onFailedChange: setHasFailed,
        refreshCookie: refreshProtectedMediaCookie,
      });
    }
    return recoveryRef.current;
  }, []);

  const copyCurrentFrame = useCallback(async () => {
    const video = localVideoRef.current;
    const accountEpoch = getAuthSession().accountEpoch;
    const result = await copyVideoFrame(
      video,
      () => localVideoRef.current === video && getAuthSession().accountEpoch === accountEpoch
    );

    if (result.ok) {
      notifications.success(t('widgets.preview.copyCurrentFrameSuccess'));
      return;
    }

    notifications.error(t(VIDEO_FRAME_COPY_FAILURE_KEYS[result.reason]));
  }, [notifications, t]);

  useEffect(() => {
    onControllerChange({ copyCurrentFrame });
    return () => {
      onControllerChange(null);
    };
  }, [copyCurrentFrame, onControllerChange]);

  useEffect(
    () => () => {
      onCopyAvailabilityChange(surfaceKey, false);
    },
    [onCopyAvailabilityChange, surfaceKey]
  );

  const handleVideoError = useCallback(() => {
    publishCopyAvailability();
    const video = localVideoRef.current;
    getRecovery().handleError(video, () => localVideoRef.current === video);
  }, [getRecovery, publishCopyAvailability]);

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
      getRecovery().retry(localVideoRef.current);
      publishCopyAvailability();
    },
    [getRecovery, publishCopyAvailability]
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

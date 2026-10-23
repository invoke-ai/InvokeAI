import { Box, Center, Flex, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, useUnavailableControl } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { VideoPlayer } from '@platform/ui/video-player/VideoPlayer';
import { getFrameAtTime, getFrameTime } from '@platform/ui/video-player/videoTimeline';
import { useCallback, useMemo, useState, type SyntheticEvent } from 'react';
import { useTranslation } from 'react-i18next';

export interface VideoFramePickerProps {
  /** The stored value, which can differ from `resolvedIndex` (a negative index, a value past the end). */
  fieldValue: number;
  fps: number;
  frameCount: number;
  onChange: (value: unknown) => void;
  posterUrl: string;
  resolvedIndex: number;
  videoUrl: string;
}

const showPausedAt = (video: HTMLVideoElement, seconds: number) => {
  video.pause();
  video.currentTime = seconds;
};

/**
 * Shows the field's frame, paused, for one mount. Keyed by the field's frame, so every change re-seeks, including ones
 * made outside this node (undo, a graph load).
 */
const FieldFrameSeek = ({ time, video }: { time: number; video: HTMLVideoElement }) => {
  useMountEffect(() => {
    const seek = () => showPausedAt(video, time);

    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      seek();

      return undefined;
    }

    video.addEventListener('loadedmetadata', seek, { once: true });

    return () => video.removeEventListener('loadedmetadata', seek);
  });

  return null;
};

/**
 * A player that follows the field: it opens on the field's frame and returns to it whenever the field changes, while
 * playing, looping and scrubbing leave the field alone. "Use this frame" commits the frame the player is paused on.
 * Key it by the source: its media state belongs to one video.
 */
export const VideoFramePicker = ({
  fieldValue,
  fps,
  frameCount,
  onChange,
  posterUrl,
  resolvedIndex,
  videoUrl,
}: VideoFramePickerProps) => {
  const { t } = useTranslation();
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  // The frame the player rests on; null while it plays or before its media loads.
  const [pausedFrame, setPausedFrame] = useState<number | null>(null);
  const [hasMediaError, setHasMediaError] = useState(false);
  const readPausedFrame = useCallback(
    ({ currentTarget }: SyntheticEvent<HTMLVideoElement>) =>
      setPausedFrame(
        currentTarget.paused && currentTarget.readyState >= HTMLMediaElement.HAVE_METADATA
          ? getFrameAtTime(currentTarget.currentTime, fps, frameCount)
          : null
      ),
    [fps, frameCount]
  );
  const handleError = useCallback(() => {
    setHasMediaError(true);
    setPausedFrame(null);
  }, []);
  const errorOverlay = useMemo(
    () =>
      hasMediaError ? (
        // A scrim: the poster stays behind the message.
        <Center bg="blackAlpha.800" inset="0" position="absolute" px="2">
          <Text color="whiteAlpha.800" fontSize="md" textAlign="center">
            {t('nodes.framePreviewPlaybackError')}
          </Text>
        </Center>
      ) : null,
    [hasMediaError, t]
  );
  const canUseFrame = pausedFrame !== null && pausedFrame !== resolvedIndex;
  // Unavailable rather than disabled: a commit makes the button unavailable while it holds focus, and a disabled
  // button would drop that focus to the page.
  const unavailable = useUnavailableControl(canUseFrame ? undefined : t('nodes.useThisFrameUnavailable'), undefined);
  // Read from the element, not the last event, so the commit is the frame on screen.
  const commitPlayerFrame = useCallback(() => {
    if (!video?.paused || video.readyState < HTMLMediaElement.HAVE_METADATA) {
      return;
    }

    const frame = getFrameAtTime(video.currentTime, fps, frameCount);

    if (frame !== resolvedIndex) {
      onChange(frame);
    }
  }, [fps, frameCount, onChange, resolvedIndex, video]);

  return (
    // `nodrag` keeps pointer drags on the scrubber from moving the node; `nokey` keeps its arrow keys from nudging it.
    <Flex className="nodrag nokey" direction="column" gap="1" w="full">
      <Box borderWidth="1px" h="32" overflow="hidden" rounded="sm" w="full">
        <VideoPlayer
          ref={setVideo}
          aria-label={t('nodes.framePreview')}
          fps={fps}
          frameCount={frameCount}
          muted
          overlay={errorOverlay}
          poster={posterUrl}
          preload="auto"
          size="compact"
          src={videoUrl}
          onError={handleError}
          onPause={readPausedFrame}
          onPlay={readPausedFrame}
          onSeeked={readPausedFrame}
        />
      </Box>
      {video ? (
        <FieldFrameSeek
          key={`${fps}|${fieldValue}|${resolvedIndex}`}
          time={getFrameTime(resolvedIndex, fps)}
          video={video}
        />
      ) : null}
      {hasMediaError ? null : (
        // Always wrapped, so the button keeps its element (and focus) as it turns unavailable.
        <Tooltip content={canUseFrame ? t('nodes.useThisFrameTooltip') : t('nodes.useThisFrameUnavailable')}>
          <Button
            alignSelf="flex-start"
            size="sm"
            variant="outline"
            {...unavailable?.props}
            onClick={commitPlayerFrame}
          >
            {t('nodes.useThisFrame')}
            {unavailable?.description}
          </Button>
        </Tooltip>
      )}
    </Flex>
  );
};

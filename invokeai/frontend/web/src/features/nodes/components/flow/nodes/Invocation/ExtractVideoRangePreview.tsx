import { CompositeSlider, Flex, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { fieldIntegerValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS } from 'features/nodes/types/constants';
import { isIntegerFieldInputInstance, isVideoFieldInputInstance } from 'features/nodes/types/field';
import { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetVideoDTOQuery } from 'services/api/endpoints/videos';

/**
 * Per-node preview for the `extract_video_range` invocation. Renders two video tiles
 * side by side — the frame at ``start_frame`` and the frame at ``end_frame`` — driven
 * by a slider underneath each tile that scrubs the field value. Negative indices
 * entered via the standard integer input are resolved against the source frame count
 * (so ``-1`` shows the last frame) while the slider always reads/writes positive
 * indices.
 *
 * The tile is just a ``<video>`` element with ``currentTime`` set to
 * ``frame_index / fps``; browsers display the seeked frame natively without us having
 * to decode to a canvas.
 */
type Props = {
  nodeId: string;
};

const ExtractVideoRangePreview = ({ nodeId }: Props) => {
  const { t } = useTranslation();
  const ctx = useInvocationNodeContext();

  const videoField = useAppSelector(useMemo(() => ctx.buildSelectInputFieldSafe('video'), [ctx]));
  const startField = useAppSelector(useMemo(() => ctx.buildSelectInputFieldSafe('start_frame'), [ctx]));
  const endField = useAppSelector(useMemo(() => ctx.buildSelectInputFieldSafe('end_frame'), [ctx]));

  const videoName = videoField && isVideoFieldInputInstance(videoField) ? videoField.value?.video_name : undefined;
  const startValue = startField && isIntegerFieldInputInstance(startField) ? startField.value : undefined;
  const endValue = endField && isIntegerFieldInputInstance(endField) ? endField.value : undefined;

  const { currentData: videoDTO } = useGetVideoDTOQuery(videoName ?? skipToken);

  if (!videoDTO) {
    return (
      <Flex mt={2} px={2} py={3} borderRadius="base" borderWidth={1} borderStyle="dashed" justifyContent="center">
        <Text fontSize="xs" color="base.400">
          {t('nodes.extractVideoRange.dropVideoPrompt')}
        </Text>
      </Flex>
    );
  }

  // VideoDTO.duration is in seconds; fps may be null for malformed uploads. Without a
  // frame count we can't drive the slider, so we fall back to a placeholder message.
  const fps = videoDTO.fps ?? null;
  const frameCount = fps && videoDTO.duration > 0 ? Math.max(1, Math.round(videoDTO.duration * fps)) : null;

  if (!fps || !frameCount) {
    return (
      <Flex mt={2} px={2} py={3} borderRadius="base" borderWidth={1} borderStyle="dashed">
        <Text fontSize="xs" color="base.400">
          {t('nodes.extractVideoRange.missingFps')}
        </Text>
      </Flex>
    );
  }

  return (
    <Flex mt={2} gap={2} className={NO_DRAG_CLASS}>
      <FrameTile
        nodeId={nodeId}
        fieldName="start_frame"
        label={t('nodes.extractVideoRange.startFrame')}
        videoUrl={videoDTO.video_url}
        rawValue={startValue}
        fps={fps}
        frameCount={frameCount}
      />
      <FrameTile
        nodeId={nodeId}
        fieldName="end_frame"
        label={t('nodes.extractVideoRange.endFrame')}
        videoUrl={videoDTO.video_url}
        rawValue={endValue}
        fps={fps}
        frameCount={frameCount}
      />
    </Flex>
  );
};

type TileProps = {
  nodeId: string;
  fieldName: 'start_frame' | 'end_frame';
  label: string;
  videoUrl: string;
  rawValue: number | undefined;
  fps: number;
  frameCount: number;
};

const FrameTile = memo(({ nodeId, fieldName, label, videoUrl, rawValue, fps, frameCount }: TileProps) => {
  const dispatch = useAppDispatch();
  const videoRef = useRef<HTMLVideoElement>(null);

  // Resolve negative indices and clamp to a valid frame so the preview never seeks
  // past the end. The integer field itself preserves the user's raw value (including
  // negative) — this only affects what the slider and <video> display.
  const resolvedIndex = useMemo(() => {
    if (rawValue === undefined) {
      return 0;
    }
    const candidate = rawValue < 0 ? frameCount + rawValue : rawValue;
    if (Number.isNaN(candidate)) {
      return 0;
    }
    return Math.max(0, Math.min(frameCount - 1, candidate));
  }, [rawValue, frameCount]);

  // Seek the video element whenever the resolved index changes. We nudge currentTime
  // by half a frame so a value of 0 doesn't sit on the keyframe boundary where some
  // codecs decode black on first paint.
  useEffect(() => {
    const el = videoRef.current;
    if (!el) {
      return;
    }
    const targetTime = (resolvedIndex + 0.5) / fps;
    const setTime = () => {
      try {
        el.currentTime = targetTime;
      } catch {
        // Seeking before metadata is available throws on some browsers — the
        // loadedmetadata listener below retries when the element is ready.
      }
    };
    if (el.readyState >= 1) {
      setTime();
    } else {
      el.addEventListener('loadedmetadata', setTime, { once: true });
      return () => el.removeEventListener('loadedmetadata', setTime);
    }
  }, [resolvedIndex, fps, videoUrl]);

  const onSliderChange = useCallback(
    (value: number) => {
      dispatch(fieldIntegerValueChanged({ nodeId, fieldName, value }));
    },
    [dispatch, fieldName, nodeId]
  );

  return (
    <Flex flex="1 1 0" flexDir="column" gap={1} minW={0}>
      <Flex justifyContent="space-between" alignItems="baseline">
        <Text fontSize="xs" color="base.300">
          {label}
        </Text>
        <Text fontSize="xs" color="base.400">
          {`${resolvedIndex} / ${frameCount - 1}`}
        </Text>
      </Flex>
      <Flex
        position="relative"
        borderRadius="base"
        borderWidth={1}
        borderStyle="solid"
        overflow="hidden"
        bg="base.900"
        h={28}
      >
        <video
          ref={videoRef}
          src={videoUrl}
          muted
          playsInline
          preload="auto"
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        />
      </Flex>
      <CompositeSlider
        value={resolvedIndex}
        onChange={onSliderChange}
        min={0}
        max={frameCount - 1}
        step={1}
        fineStep={1}
        defaultValue={fieldName === 'start_frame' ? 0 : frameCount - 1}
        withThumbTooltip
      />
    </Flex>
  );
});

FrameTile.displayName = 'FrameTile';

export default memo(ExtractVideoRangePreview);

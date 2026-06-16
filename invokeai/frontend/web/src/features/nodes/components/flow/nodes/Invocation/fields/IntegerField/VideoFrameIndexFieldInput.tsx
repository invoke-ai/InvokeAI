import { CompositeNumberInput, CompositeSlider, Flex, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { useIntegerField } from 'features/nodes/components/flow/nodes/Invocation/fields/IntegerField/useIntegerField';
import { selectFieldInputInstanceSafe, selectNodesSlice } from 'features/nodes/store/selectors';
import { NO_DRAG_CLASS } from 'features/nodes/types/constants';
import type { IntegerFieldInputInstance, IntegerFieldInputTemplate } from 'features/nodes/types/field';
import { isVideoFieldInputInstance } from 'features/nodes/types/field';
import { memo, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetVideoDTOQuery } from 'services/api/endpoints/videos';

/**
 * Integer-field renderer used by ``start_frame`` and ``end_frame`` on the
 * ``extract_video_range`` invocation. Renders the standard number input plus a
 * live frame thumbnail and a scrubber slider, all writing to the same Redux
 * integer field. The thumbnail is a ``<video>`` element seeked to
 * ``frame / fps``; browsers display the frame natively without a canvas
 * roundtrip.
 *
 * The widget looks up its companion ``VideoField`` directly via Redux on the
 * same node, so it works wherever ``InputFieldRenderer`` is used — the
 * workflow editor's node body AND the Form Builder's view/edit modes.
 *
 * Convention: the source video is expected to live on a sibling field named
 * ``video`` on the same node. If that field is missing or unset, the widget
 * gracefully falls back to a plain number input.
 */
const COMPANION_VIDEO_FIELD_NAME = 'video';

export const VideoFrameIndexFieldInput = memo(
  (props: FieldComponentProps<IntegerFieldInputInstance, IntegerFieldInputTemplate>) => {
    const { t } = useTranslation();
    const { nodeId, field, fieldTemplate } = props;
    const { defaultValue, onChange, min, max, step, fineStep, constrainValue } = useIntegerField(
      nodeId,
      field.name,
      fieldTemplate
    );

    const selectVideoName = useMemo(
      () =>
        createSelector(selectNodesSlice, (nodes) => {
          const sibling = selectFieldInputInstanceSafe(nodes, nodeId, COMPANION_VIDEO_FIELD_NAME);
          if (!sibling || !isVideoFieldInputInstance(sibling)) {
            return undefined;
          }
          return sibling.value?.video_name;
        }),
      [nodeId]
    );
    const videoName = useAppSelector(selectVideoName);
    const { currentData: videoDTO } = useGetVideoDTOQuery(videoName ?? skipToken);

    // Frame count is the slider's upper bound. duration*fps can be off-by-one for VFR
    // containers, but that's tolerable here — the slider is for visual scrubbing, not
    // for the authoritative range check (which the backend re-validates at invoke time).
    const fps = videoDTO?.fps ?? null;
    const frameCount =
      videoDTO && fps && videoDTO.duration > 0 ? Math.max(1, Math.round(videoDTO.duration * fps)) : null;

    // Resolve negative indices (e.g. -1 = last frame) for display only — the underlying
    // field value is preserved verbatim so users can still type "-1" and have the
    // backend resolve it at invoke time against the authoritative decoder frame count.
    const resolvedIndex = useMemo(() => {
      if (frameCount === null || field.value === undefined) {
        return 0;
      }
      const candidate = field.value < 0 ? frameCount + field.value : field.value;
      if (Number.isNaN(candidate)) {
        return 0;
      }
      return Math.max(0, Math.min(frameCount - 1, candidate));
    }, [field.value, frameCount]);

    return (
      <Flex flexDir="column" gap={1} w="full" className={NO_DRAG_CLASS}>
        <CompositeNumberInput
          defaultValue={defaultValue}
          onChange={onChange}
          value={field.value}
          min={min}
          max={max}
          step={step}
          fineStep={fineStep}
          flex="1 1 0"
          constrainValue={constrainValue}
          allowMath
        />
        {videoDTO && frameCount && fps ? (
          <FrameScrubber
            videoUrl={videoDTO.video_url}
            resolvedIndex={resolvedIndex}
            fps={fps}
            frameCount={frameCount}
            onChange={onChange}
          />
        ) : (
          <Flex mt={1} px={2} py={2} borderRadius="base" borderWidth={1} borderStyle="dashed" justifyContent="center">
            <Text fontSize="xs" color="base.400">
              {videoName ? t('nodes.extractVideoRange.missingFps') : t('nodes.extractVideoRange.dropVideoPrompt')}
            </Text>
          </Flex>
        )}
      </Flex>
    );
  }
);

VideoFrameIndexFieldInput.displayName = 'VideoFrameIndexFieldInput';

type FrameScrubberProps = {
  videoUrl: string;
  resolvedIndex: number;
  fps: number;
  frameCount: number;
  onChange: (value: number) => void;
};

const FrameScrubber = memo(({ videoUrl, resolvedIndex, fps, frameCount, onChange }: FrameScrubberProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Seek the video element whenever the resolved index changes. We nudge currentTime
  // by half a frame so the seek lands inside the frame's display window — some codecs
  // decode the boundary as black on first paint without this offset.
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

  return (
    <Flex flexDir="column" gap={1}>
      <Flex
        position="relative"
        borderRadius="base"
        borderWidth={1}
        borderStyle="solid"
        overflow="hidden"
        bg="base.900"
        h={32}
      >
        <video
          ref={videoRef}
          src={videoUrl}
          muted
          playsInline
          preload="auto"
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        />
        <Text
          position="absolute"
          insetInlineEnd={1}
          insetBlockEnd={1}
          background="base.900"
          color="base.50"
          fontSize="xs"
          fontWeight="semibold"
          opacity={0.7}
          px={2}
          borderRadius="base"
          pointerEvents="none"
        >
          {`${resolvedIndex} / ${frameCount - 1}`}
        </Text>
      </Flex>
      <CompositeSlider
        value={resolvedIndex}
        onChange={onChange}
        min={0}
        max={frameCount - 1}
        step={1}
        fineStep={1}
        defaultValue={0}
        withThumbTooltip
      />
    </Flex>
  );
});

FrameScrubber.displayName = 'FrameScrubber';

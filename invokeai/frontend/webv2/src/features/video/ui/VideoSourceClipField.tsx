import type { GalleryItem } from '@features/gallery';
import type { VideoSourceClip } from '@features/video/core/types';

import { HStack, Stack, Text } from '@chakra-ui/react';
import { GalleryMediaSlot, type GalleryMediaSlotLabels, type GalleryMediaSlotValue } from '@features/gallery/mediaSlot';
import { galleryVideoUrls } from '@features/gallery/utility';
import { createVideoSourceClip } from '@features/video/core/settings';
import { Field } from '@platform/ui';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { PlayClipSpanButton } from './PlayClipSpanButton';
import { TrimBoundThumb } from './TrimBoundThumb';
import { useVideoUiActions } from './VideoUiContext';

const DROP_ID = 'video-source-clip';
const VIDEO_ONLY = ['video'] as const;

const areSourceClipsEquivalent = (left: VideoSourceClip | null, right: VideoSourceClip | null): boolean =>
  (left === null && right === null) ||
  (left?.video_name === right?.video_name &&
    left?.startFrame === right?.startFrame &&
    left?.endFrame === right?.endFrame &&
    // The bound thumbnails seek at frame/fps, so a same-name clip hydrated
    // with a different probed rate must re-render.
    left?.fps === right?.fps);

export const VideoSourceClipField = memo(
  function VideoSourceClipField({
    disabled = false,
    disabledReason,
    onChange,
    sourceVideo,
  }: {
    disabled?: boolean;
    /** Shown in place of the field's affordances while disabled (mutual exclusion). */
    disabledReason?: string;
    onChange: (clip: VideoSourceClip | null) => void;
    sourceVideo: VideoSourceClip | null;
  }) {
    const { t } = useTranslation();
    const { findInGallery } = useVideoUiActions();
    const slotLabels = useMemo<Partial<GalleryMediaSlotLabels>>(
      () => ({ drop: t('widgets.video.dropInitialVideo') }),
      [t]
    );
    const slotValue = useMemo<GalleryMediaSlotValue | null>(
      () =>
        sourceVideo
          ? { height: sourceVideo.height, kind: 'video', name: sourceVideo.video_name, width: sourceVideo.width }
          : null,
      [sourceVideo]
    );
    const handleSlotChange = useCallback(
      (item: GalleryItem | null) => {
        if (item === null) {
          onChange(null);
        } else if (item.kind === 'video') {
          onChange(createVideoSourceClip(item));
        }
      },
      [onChange]
    );

    const maxFrameIndex = sourceVideo ? Math.max(0, sourceVideo.numFrames - 1) : 0;
    // The crossfade join needs a 2-frame tail from the trimmed source, so the
    // setters keep at least two frames between the bounds.
    const setStartFrame = useCallback(
      (rawStart: number) => {
        if (!sourceVideo) {
          return;
        }

        const startFrame = Math.min(Math.max(0, rawStart), Math.max(0, maxFrameIndex - 1));

        onChange({ ...sourceVideo, endFrame: Math.max(startFrame + 1, sourceVideo.endFrame), startFrame });
      },
      [maxFrameIndex, onChange, sourceVideo]
    );
    const setEndFrame = useCallback(
      (rawEnd: number) => {
        if (!sourceVideo) {
          return;
        }

        const endFrame = Math.max(1, Math.min(rawEnd, maxFrameIndex));

        onChange({ ...sourceVideo, endFrame, startFrame: Math.min(sourceVideo.startFrame, endFrame - 1) });
      },
      [maxFrameIndex, onChange, sourceVideo]
    );

    const previewSrc = sourceVideo ? galleryVideoUrls.full(sourceVideo.video_name) : null;
    // Both bounds reference one gallery record; expose one find control.
    const videoName = sourceVideo?.video_name;
    const findClipInGallery = useCallback(() => {
      if (videoName !== undefined) {
        findInGallery({ kind: 'video', name: videoName });
      }
    }, [findInGallery, videoName]);

    return (
      <Stack gap="2">
        <GalleryMediaSlot
          accept={VIDEO_ONLY}
          disabled={disabled}
          disabledReason={disabledReason}
          dropId={DROP_ID}
          labels={slotLabels}
          value={slotValue}
          onChange={handleSlotChange}
        />
        {sourceVideo && previewSrc ? (
          <Stack gap="2">
            {/* Show the disabled reason even when a clip is already set. */}
            {disabled && disabledReason ? (
              <Text color="fg.muted" fontSize="2xs" textWrap="pretty">
                {disabledReason}
              </Text>
            ) : null}
            <Field helpText={t('widgets.video.trimHelp')} label={t('widgets.video.trim')}>
              <HStack align="center" gap="2" w="full">
                <PlayClipSpanButton clip={sourceVideo} />
                <Stack flex="1" gap="1" minW="0">
                  <HStack gap="2">
                    <TrimBoundThumb
                      fps={sourceVideo.fps}
                      frame={sourceVideo.startFrame}
                      label={t('widgets.video.trimStartShort')}
                      name={sourceVideo.video_name}
                      src={previewSrc}
                      onFindInGallery={findClipInGallery}
                    />
                    <ScrubberField
                      disabled={disabled}
                      label={t('widgets.video.trimStart')}
                      max={maxFrameIndex}
                      min={0}
                      step={1}
                      value={sourceVideo.startFrame}
                      onChange={setStartFrame}
                    />
                  </HStack>
                  <HStack gap="2">
                    <TrimBoundThumb
                      fps={sourceVideo.fps}
                      frame={sourceVideo.endFrame}
                      label={t('widgets.video.trimEndShort')}
                      src={previewSrc}
                    />
                    <ScrubberField
                      disabled={disabled}
                      label={t('widgets.video.trimEnd')}
                      max={maxFrameIndex}
                      min={0}
                      step={1}
                      value={sourceVideo.endFrame}
                      onChange={setEndFrame}
                    />
                  </HStack>
                </Stack>
              </HStack>
            </Field>
          </Stack>
        ) : null}
      </Stack>
    );
  },
  (previous, next) =>
    previous.onChange === next.onChange &&
    previous.disabled === next.disabled &&
    previous.disabledReason === next.disabledReason &&
    areSourceClipsEquivalent(previous.sourceVideo, next.sourceVideo)
);

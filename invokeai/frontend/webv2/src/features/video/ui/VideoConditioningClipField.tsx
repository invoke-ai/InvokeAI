import type { GalleryItem } from '@features/gallery';
import type { VideoConditioningClip, VideoConditioningRole } from '@features/video/core/types';

import { createListCollection, Stack, Text } from '@chakra-ui/react';
import { GalleryMediaSlot, type GalleryMediaSlotLabels, type GalleryMediaSlotValue } from '@features/gallery/mediaSlot';
import { createVideoConditioningClip } from '@features/video/core/settings';
import { Field, Select } from '@platform/ui';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useVideoUiActions } from './VideoUiContext';

/**
 * The whole-modality conditioning slot: a clip whose soundtrack the model generates a picture for,
 * or whose picture it generates a soundtrack for. One slot with a role, not two slots, because
 * LTX-2 holds exactly one modality clean and samples the other.
 *
 * No trim bounds: the conditioning nodes consume the whole recording.
 */

const DROP_ID = 'video-conditioning-clip';
const VIDEO_ONLY = ['video'] as const;

export const VideoConditioningClipField = memo(
  function VideoConditioningClipField({
    conditioningClip,
    disabled = false,
    disabledReason,
    derivedText,
    onChange,
  }: {
    conditioningClip: VideoConditioningClip | null;
    disabled?: boolean;
    /** Shown in place of the field's affordances while disabled (mutual exclusion). */
    disabledReason?: string;
    /** What the clip works out to — length, and the rate it will run at. */
    derivedText?: string;
    onChange: (conditioning: VideoConditioningClip | null) => void;
  }) {
    const { t } = useTranslation();
    const { findInGallery } = useVideoUiActions();
    const slotLabels = useMemo<Partial<GalleryMediaSlotLabels>>(
      () => ({ drop: t('widgets.video.dropConditioningClip') }),
      [t]
    );
    const slotValue = useMemo<GalleryMediaSlotValue | null>(
      () =>
        conditioningClip
          ? {
              height: conditioningClip.clip.height,
              kind: 'video',
              name: conditioningClip.clip.video_name,
              width: conditioningClip.clip.width,
            }
          : null,
      [conditioningClip]
    );
    const roleCollection = useMemo(
      () =>
        createListCollection({
          items: [
            { label: t('widgets.video.conditioningRoleAudio'), value: 'audio' },
            { label: t('widgets.video.conditioningRoleVideo'), value: 'video' },
          ],
        }),
      [t]
    );
    const roleValue = useMemo(() => (conditioningClip ? [conditioningClip.role] : []), [conditioningClip]);

    const handleSlotChange = useCallback(
      (item: GalleryItem | null) => {
        if (item === null) {
          onChange(null);
        } else if (item.kind === 'video') {
          onChange(createVideoConditioningClip(item));
        }
      },
      [onChange]
    );
    // The whole premise of this field is what is inside the clip, so it gets the same way back to
    // the gallery record that every other media slot has.
    const videoName = conditioningClip?.clip.video_name;
    const handleFind = useCallback(() => {
      if (videoName !== undefined) {
        findInGallery({ kind: 'video', name: videoName });
      }
    }, [findInGallery, videoName]);
    const handleRoleChange = useCallback(
      ({ value }: { value: string[] }) => {
        const role = value[0];

        if (conditioningClip && (role === 'audio' || role === 'video')) {
          onChange({ ...conditioningClip, role: role satisfies VideoConditioningRole });
        }
      },
      [conditioningClip, onChange]
    );

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
          onFind={handleFind}
        />
        {conditioningClip ? (
          <Field
            helpText={
              conditioningClip.role === 'video'
                ? `${t('widgets.video.conditioningRoleVideoHelp')}${derivedText ? ` ${derivedText}` : ''}`
                : derivedText
            }
            label={t('widgets.video.conditioningRole')}
          >
            <Select
              collection={roleCollection}
              disabled={disabled}
              size="xs"
              value={roleValue}
              onValueChange={handleRoleChange}
            />
          </Field>
        ) : (
          <Text color="fg.muted" fontSize="2xs" textWrap="pretty">
            {t('widgets.video.conditioningClipHelp')}
          </Text>
        )}
      </Stack>
    );
  },
  (previous, next) =>
    previous.onChange === next.onChange &&
    previous.disabled === next.disabled &&
    previous.disabledReason === next.disabledReason &&
    previous.derivedText === next.derivedText &&
    previous.conditioningClip?.clip.video_name === next.conditioningClip?.clip.video_name &&
    previous.conditioningClip?.role === next.conditioningClip?.role
);

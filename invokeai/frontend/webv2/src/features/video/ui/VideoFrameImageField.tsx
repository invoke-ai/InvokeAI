import type { GalleryItem } from '@features/gallery';
import type { GalleryMediaSlotLabels, GalleryMediaSlotValue } from '@features/gallery/mediaSlot';
import type { ImageWithDims } from '@features/generation/contracts';

import { GalleryMediaSlot } from '@features/gallery/mediaSlot';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useVideoUiActions } from './VideoUiContext';

const ACCEPT = ['image'] as const;

const areFrameImagesEquivalent = (left: ImageWithDims | null, right: ImageWithDims | null): boolean =>
  (left === null && right === null) || left?.image_name === right?.image_name;

/** A conditioning keyframe (first or last frame): pick from the gallery, drop from it, or upload. */
export const VideoFrameImageField = memo(
  function VideoFrameImageField({
    disabled = false,
    disabledReason,
    dropId,
    dropLabel,
    slotLabel,
    image,
    onChange,
  }: {
    disabled?: boolean;
    /** Shown in place of the field's affordances while disabled (mutual exclusion). */
    disabledReason?: string;
    /** Unique droppable id — the first- and last-frame fields render side by side. */
    dropId: string;
    /** Call-to-action shown while a compatible drag is in flight ("Drop First Frame"). */
    dropLabel: string;
    slotLabel?: string;
    image: ImageWithDims | null;
    onChange: (image: ImageWithDims | null) => void;
  }) {
    const { t } = useTranslation();
    const { findInGallery } = useVideoUiActions();
    const value = useMemo<GalleryMediaSlotValue | null>(
      () => (image ? { height: image.height, kind: 'image', name: image.image_name, width: image.width } : null),
      [image]
    );
    const labels = useMemo<Partial<GalleryMediaSlotLabels>>(
      () => ({
        drop: dropLabel,
        ...(slotLabel
          ? {
              choose: t('widgets.video.hosted.chooseSlot', { slot: slotLabel }),
              replace: t('widgets.video.hosted.replaceSlot', { slot: slotLabel, name: image?.image_name }),
              remove: t('widgets.video.hosted.removeSlot', { slot: slotLabel }),
            }
          : {}),
      }),
      [dropLabel, image?.image_name, slotLabel, t]
    );
    const handleChange = useCallback(
      (item: GalleryItem | null) =>
        onChange(item?.kind === 'image' ? { height: item.height, image_name: item.name, width: item.width } : null),
      [onChange]
    );
    const imageName = image?.image_name;
    const handleFind = useCallback(() => {
      if (imageName !== undefined) {
        findInGallery({ kind: 'image', name: imageName });
      }
    }, [findInGallery, imageName]);

    return (
      <GalleryMediaSlot
        accept={ACCEPT}
        disabled={disabled}
        disabledReason={disabledReason}
        dropId={dropId}
        labels={labels}
        value={value}
        onChange={handleChange}
        onFind={handleFind}
      />
    );
  },
  (previous, next) =>
    previous.onChange === next.onChange &&
    previous.disabled === next.disabled &&
    previous.disabledReason === next.disabledReason &&
    previous.dropId === next.dropId &&
    previous.dropLabel === next.dropLabel &&
    previous.slotLabel === next.slotLabel &&
    areFrameImagesEquivalent(previous.image, next.image)
);

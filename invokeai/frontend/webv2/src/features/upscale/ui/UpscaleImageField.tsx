import type { GalleryItem } from '@features/gallery';
import type { GalleryMediaSlotValue } from '@features/gallery/mediaSlot';
import type { UpscaleWidgetValues } from '@features/upscale/core/types';

import { GalleryMediaSlot } from '@features/gallery/mediaSlot';
import { memo, useCallback, useMemo } from 'react';

import { areInputImagesEquivalent } from './upscaleComparators';

const DROP_ID = 'upscale-input-image';
const ACCEPT = ['image'] as const;

/** The Upscale widget's source image: pick from the gallery, drop from it, or upload. */
export const UpscaleImageField = memo(
  function UpscaleImageField({
    inputImage,
    onChange,
  }: {
    inputImage: UpscaleWidgetValues['inputImage'];
    onChange: (image: UpscaleWidgetValues['inputImage']) => void;
  }) {
    const value = useMemo<GalleryMediaSlotValue | null>(
      () =>
        inputImage
          ? { height: inputImage.height, kind: 'image', name: inputImage.image_name, width: inputImage.width }
          : null,
      [inputImage]
    );
    const handleChange = useCallback(
      (item: GalleryItem | null) =>
        onChange(item?.kind === 'image' ? { height: item.height, image_name: item.name, width: item.width } : null),
      [onChange]
    );

    return <GalleryMediaSlot accept={ACCEPT} dropId={DROP_ID} value={value} onChange={handleChange} />;
  },
  (previous, next) =>
    previous.onChange === next.onChange && areInputImagesEquivalent(previous.inputImage, next.inputImage)
);

import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { SelectForCompareDropData } from 'features/dnd/types';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(selectGallerySlice, (gallerySlice) => {
  const firstImage = gallerySlice.selection.slice(-1)[0] ?? null;
  const secondImage = gallerySlice.imageToCompare;
  return { firstImage, secondImage };
});

export const ImageComparisonDroppable = memo(() => {
  const { t } = useTranslation();
  const { firstImage, secondImage } = useAppSelector(selector);
  const droppableData = useMemo<SelectForCompareDropData>(
    () => ({
      id: 'image-comparison',
      actionType: 'SELECT_FOR_COMPARE',
      context: {
        firstImageName: firstImage?.image_name,
        secondImageName: secondImage?.image_name,
      },
    }),
    [firstImage?.image_name, secondImage?.image_name]
  );

  return <IAIDroppable data={droppableData} dropLabel={t('gallery.selectForCompare')} />;
});

ImageComparisonDroppable.displayName = 'ImageComparisonDroppable';

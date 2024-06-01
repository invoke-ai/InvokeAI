import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { CurrentImageDropData, SelectForCompareDropData } from 'features/dnd/types';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(selectGallerySlice, (gallerySlice) => {
  const firstImage = gallerySlice.selection.slice(-1)[0] ?? null;
  const secondImage = gallerySlice.imageToCompare;
  return { firstImage, secondImage };
});

const setCurrentImageDropData: CurrentImageDropData = {
  id: 'current-image',
  actionType: 'SET_CURRENT_IMAGE',
};

export const ImageComparisonDroppable = memo(() => {
  const { t } = useTranslation();
  const { firstImage, secondImage } = useAppSelector(selector);
  const selectForCompareDropData = useMemo<SelectForCompareDropData>(
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

  return (
    <Flex position="absolute" top={0} right={0} bottom={0} left={0} gap={2} pointerEvents="none">
      <Flex position="relative" flex={1}>
        <IAIDroppable data={setCurrentImageDropData} dropLabel={t('gallery.selectForViewer')} />
      </Flex>
      <Flex position="relative" flex={1}>
        <IAIDroppable data={selectForCompareDropData} dropLabel={t('gallery.selectForCompare')} />
      </Flex>
    </Flex>
  );
});

ImageComparisonDroppable.displayName = 'ImageComparisonDroppable';

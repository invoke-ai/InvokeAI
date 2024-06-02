import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { CurrentImageDropData, SelectForCompareDropData } from 'features/dnd/types';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { selectComparisonImages } from './common';

const setCurrentImageDropData: CurrentImageDropData = {
  id: 'current-image',
  actionType: 'SET_CURRENT_IMAGE',
};

export const ImageComparisonDroppable = memo(() => {
  const { t } = useTranslation();
  const imageViewer = useImageViewer();
  const { firstImage, secondImage } = useAppSelector(selectComparisonImages);
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

  if (!imageViewer.isOpen) {
    return (
      <Flex position="absolute" top={0} right={0} bottom={0} left={0} gap={2} pointerEvents="none">
        <IAIDroppable data={setCurrentImageDropData} dropLabel={t('gallery.openInViewer')} />
      </Flex>
    );
  }

  return (
    <Flex position="absolute" top={0} right={0} bottom={0} left={0} gap={2} pointerEvents="none">
      <IAIDroppable data={selectForCompareDropData} dropLabel={t('gallery.selectForCompare')} />
    </Flex>
  );
});

ImageComparisonDroppable.displayName = 'ImageComparisonDroppable';

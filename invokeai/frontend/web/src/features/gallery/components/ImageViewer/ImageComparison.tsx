import type { UseMeasureRect } from '@reactuses/core';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { SelectForCompareDropData } from 'features/dnd/types';
import { ImageComparisonSideBySide } from 'features/gallery/components/ImageViewer/ImageComparisonSideBySide';
import { ImageComparisonSlider } from 'features/gallery/components/ImageViewer/ImageComparisonSlider';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = {
  containerSize: UseMeasureRect;
};

export const ImageComparison = memo(({ containerSize }: Props) => {
  const comparisonMode = useAppSelector((s) => s.gallery.comparisonMode);
  const { firstImage, secondImage } = useAppSelector((s) => {
    const firstImage = s.gallery.selection.slice(-1)[0] ?? null;
    const secondImage = s.gallery.imageToCompare;
    return { firstImage, secondImage };
  });

  if (!firstImage || !secondImage) {
    return <ImageComparisonWrapper>No images to compare</ImageComparisonWrapper>;
  }

  if (comparisonMode === 'slider') {
    return (
      <ImageComparisonWrapper>
        <ImageComparisonSlider containerSize={containerSize} firstImage={firstImage} secondImage={secondImage} />
      </ImageComparisonWrapper>
    );
  }

  if (comparisonMode === 'side-by-side') {
    return (
      <ImageComparisonWrapper>
        <ImageComparisonSideBySide firstImage={firstImage} secondImage={secondImage} />
      </ImageComparisonWrapper>
    );
  }
});

ImageComparison.displayName = 'ImageComparison';

const droppableData: SelectForCompareDropData = {
  id: 'image-comparison',
  actionType: 'SELECT_FOR_COMPARE',
};

const ImageComparisonWrapper = memo((props: PropsWithChildren) => {
  return (
    <>
      {props.children}
      <IAIDroppable data={droppableData} dropLabel="Select for Compare" />
    </>
  );
});

ImageComparisonWrapper.displayName = 'ImageComparisonWrapper';

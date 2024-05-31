import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import type { SelectForCompareDropData } from 'features/dnd/types';
import { ImageComparisonSideBySide } from 'features/gallery/components/ImageViewer/ImageComparisonSideBySide';
import { ImageComparisonSlider } from 'features/gallery/components/ImageViewer/ImageComparisonSlider';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

const selector = createMemoizedSelector(selectGallerySlice, (gallerySlice) => {
  const firstImage = gallerySlice.selection.slice(-1)[0] ?? null;
  const secondImage = gallerySlice.imageToCompare;
  return { firstImage, secondImage };
});

export const ImageComparison = memo(() => {
  const comparisonMode = useAppSelector((s) => s.gallery.comparisonMode);
  const { firstImage, secondImage } = useAppSelector(selector);

  if (!firstImage || !secondImage) {
    return <ImageComparisonWrapper>Select an image to compare</ImageComparisonWrapper>;
  }

  if (comparisonMode === 'slider') {
    return (
      <ImageComparisonWrapper>
        <ImageComparisonSlider firstImage={firstImage} secondImage={secondImage} />
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

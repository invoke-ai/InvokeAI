import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDroppable from 'common/components/IAIDroppable';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import type { SelectForCompareDropData } from 'features/dnd/types';
import { ImageComparisonSideBySide } from 'features/gallery/components/ImageViewer/ImageComparisonSideBySide';
import { ImageComparisonSlider } from 'features/gallery/components/ImageViewer/ImageComparisonSlider';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

const selector = createMemoizedSelector(selectGallerySlice, (gallerySlice) => {
  const firstImage = gallerySlice.selection.slice(-1)[0] ?? null;
  const secondImage = gallerySlice.imageToCompare;
  return { firstImage, secondImage };
});

export const ImageComparison = memo(() => {
  const { t } = useTranslation();
  const comparisonMode = useAppSelector((s) => s.gallery.comparisonMode);
  const { firstImage, secondImage } = useAppSelector(selector);

  if (!firstImage || !secondImage) {
    return (
      <ImageComparisonWrapper firstImage={firstImage} secondImage={secondImage}>
        <IAINoContentFallback label={t('gallery.selectAnImageToCompare')} icon={PiImagesBold} />
      </ImageComparisonWrapper>
    );
  }

  if (comparisonMode === 'slider') {
    return (
      <ImageComparisonWrapper firstImage={firstImage} secondImage={secondImage}>
        <ImageComparisonSlider firstImage={firstImage} secondImage={secondImage} />
      </ImageComparisonWrapper>
    );
  }

  if (comparisonMode === 'side-by-side') {
    return (
      <ImageComparisonWrapper firstImage={firstImage} secondImage={secondImage}>
        <ImageComparisonSideBySide firstImage={firstImage} secondImage={secondImage} />
      </ImageComparisonWrapper>
    );
  }
});

ImageComparison.displayName = 'ImageComparison';

type Props = PropsWithChildren<{
  firstImage: ImageDTO | null;
  secondImage: ImageDTO | null;
}>;

const ImageComparisonWrapper = memo((props: Props) => {
  const droppableData = useMemo<SelectForCompareDropData>(
    () => ({
      id: 'image-comparison',
      actionType: 'SELECT_FOR_COMPARE',
      context: {
        firstImageName: props.firstImage?.image_name,
        secondImageName: props.secondImage?.image_name,
      },
    }),
    [props.firstImage?.image_name, props.secondImage?.image_name]
  );

  return (
    <>
      {props.children}
      <IAIDroppable data={droppableData} dropLabel="Select for Compare" />
    </>
  );
});

ImageComparisonWrapper.displayName = 'ImageComparisonWrapper';

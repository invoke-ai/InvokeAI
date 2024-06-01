import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { ImageComparisonSideBySide } from 'features/gallery/components/ImageViewer/ImageComparisonSideBySide';
import { ImageComparisonSlider } from 'features/gallery/components/ImageViewer/ImageComparisonSlider';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesBold } from 'react-icons/pi';

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
    // Should rarely/never happen - we don't render this component unless we have images to compare
    return <IAINoContentFallback label={t('gallery.selectAnImageToCompare')} icon={PiImagesBold} />;
  }

  if (comparisonMode === 'slider') {
    return <ImageComparisonSlider firstImage={firstImage} secondImage={secondImage} />;
  }

  if (comparisonMode === 'side-by-side') {
    return <ImageComparisonSideBySide firstImage={firstImage} secondImage={secondImage} />;
  }
});

ImageComparison.displayName = 'ImageComparison';

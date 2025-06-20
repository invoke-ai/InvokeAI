import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectImageToCompare } from 'features/gallery/components/ImageViewer/common';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageComparison } from 'features/gallery/components/ImageViewer/ImageComparison';
import { selectLastSelectedImageName } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

export const ImageViewer = memo(() => {
  const lastSelectedImageName = useAppSelector(selectLastSelectedImageName);
  const { data: lastSelectedImageDTO } = useGetImageDTOQuery(lastSelectedImageName ?? skipToken);
  const comparisonImageDTO = useAppSelector(selectImageToCompare);

  if (lastSelectedImageDTO && comparisonImageDTO) {
    return <ImageComparison firstImage={lastSelectedImageDTO} secondImage={comparisonImageDTO} />;
  }

  return <CurrentImagePreview imageDTO={lastSelectedImageDTO} />;
});

ImageViewer.displayName = 'ImageViewer';

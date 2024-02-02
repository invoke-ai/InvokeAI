import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const selectLastSelectedImageName = createSelector(
  selectLastSelectedImage,
  (lastSelectedImage) => lastSelectedImage?.image_name
);

export const ViewerInfo = memo(() => {
  const { t } = useTranslation();
  const imageName = useAppSelector(selectLastSelectedImageName);
  const { currentData: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);

  if (!imageDTO) {
    return <IAINoContentFallback icon={PiImageBold} label={t('gallery.noImageSelected')} />;
  }

  return <ImageMetadataViewer image={imageDTO} />;
});

ViewerInfo.displayName = 'ViewerInfo';

import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import type { TypesafeDraggableData } from 'features/dnd/types';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const selectLastSelectedImageName = createSelector(
  selectLastSelectedImage,
  (lastSelectedImage) => lastSelectedImage?.image_name
);

export const ViewerImage = memo(() => {
  const imageName = useAppSelector(selectLastSelectedImageName);

  const { currentData: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (imageDTO) {
      return {
        id: 'current-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO]);

  const { t } = useTranslation();

  return (
    <IAIDndImage
      imageDTO={imageDTO}
      draggableData={draggableData}
      isUploadDisabled={true}
      fitContainer
      fallbackSrc={imageDTO?.thumbnail_url}
      noContentFallback={<IAINoContentFallback icon={PiImageBold} label={t('gallery.noImageSelected')} />}
      dataTestId="image-preview"
    />
  );
});

ViewerImage.displayName = 'ViewerImage';

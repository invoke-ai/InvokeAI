import { Box, Flex, Grid } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { useGalleryHotkeys } from 'features/gallery/hooks/useGalleryHotkeys';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold, PiWarningCircleBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import GalleryImage from './GalleryImage';
import { useListImagesQuery } from '../../../../services/api/endpoints/images';
import { GalleryPagination } from './GalleryPagination';

export const imageListContainerTestId = 'image-list-container';
export const imageItemContainerTestId = 'image-item-container';

const GalleryImageGrid = () => {
  useGalleryHotkeys();
  const { t } = useTranslation();
  const galleryImageMinimumWidth = useAppSelector((s) => s.gallery.galleryImageMinimumWidth);
  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const { imageDTOs, isLoading, isSuccess, isError } = useListImagesQuery(queryArgs, {
    selectFromResult: ({ data, isLoading, isSuccess, isError }) => ({
      imageDTOs: data?.items ?? EMPTY_ARRAY,
      isLoading,
      isSuccess,
      isError,
    }),
  });

  if (isError) {
    return (
      <Box w="full" h="full">
        <IAINoContentFallback label={t('gallery.unableToLoad')} icon={PiWarningCircleBold} />
      </Box>
    );
  }

  if (isLoading) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center">
        <IAINoContentFallback label={t('gallery.loading')} icon={PiImageBold} />
      </Flex>
    );
  }

  if (imageDTOs.length === 0) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center">
        <IAINoContentFallback label={t('gallery.noImagesInGallery')} icon={PiImageBold} />
      </Flex>
    );
  }

  return (
    <>
      <Box data-overlayscrollbars="" h="100%" id="gallery-grid">
        <Grid
          className="list-container"
          gridTemplateColumns={`repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr))`}
          data-testid={imageListContainerTestId}
        >
          {imageDTOs.map((imageDTO, index) => (
            <GalleryImage key={imageDTO.image_name} imageDTO={imageDTO} index={index} />
          ))}
        </Grid>
      </Box>
      <GalleryPagination />
    </>
  );
};

export default memo(GalleryImageGrid);

const GalleryImageContainer = memo(({ imageDTO, index }: { imageDTO: ImageDTO; index: number }) => {
  return (
    <Box className="item-container" p={1.5} data-testid={imageItemContainerTestId}>
      <GalleryImage imageDTO={imageDTO} index={index} />
    </Box>
  );
});

GalleryImageContainer.displayName = 'GalleryImageContainer';

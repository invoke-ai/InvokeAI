import { Box, Flex, Grid, IconButton } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { useGalleryHotkeys } from 'features/gallery/hooks/useGalleryHotkeys';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryImages';
import { selectListImages2QueryArgs } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCaretDoubleLeftBold,
  PiCaretDoubleRightBold,
  PiCaretLeftBold,
  PiCaretRightBold,
  PiImageBold,
  PiWarningCircleBold,
} from 'react-icons/pi';
import { useListImages2Query } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

import GalleryImage from './GalleryImage';

export const imageListContainerTestId = 'image-list-container';
export const imageItemContainerTestId = 'image-item-container';

const GalleryImageGrid = () => {
  useGalleryHotkeys();
  const { t } = useTranslation();
  const galleryImageMinimumWidth = useAppSelector((s) => s.gallery.galleryImageMinimumWidth);
  const queryArgs = useAppSelector(selectListImages2QueryArgs);
  const { imageDTOs, isLoading, isSuccess, isError } = useListImages2Query(queryArgs, {
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

const GalleryPagination = memo(() => {
  const { first, prev, next, last, isFirstEnabled, isPrevEnabled, isNextEnabled, isLastEnabled } =
    useGalleryPagination();
  return (
    <Flex gap={2}>
      <IconButton aria-label="prev" icon={<PiCaretDoubleLeftBold />} onClick={first} isDisabled={!isFirstEnabled} />
      <IconButton aria-label="prev" icon={<PiCaretLeftBold />} onClick={prev} isDisabled={!isPrevEnabled} />
      <IconButton aria-label="next" icon={<PiCaretRightBold />} onClick={next} isDisabled={!isNextEnabled} />
      <IconButton aria-label="next" icon={<PiCaretDoubleRightBold />} onClick={last} isDisabled={!isLastEnabled} />
    </Flex>
  );
});

GalleryPagination.displayName = 'GalleryPagination';

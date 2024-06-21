import { Box, Flex, Grid } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { useGalleryHotkeys } from 'features/gallery/hooks/useGalleryHotkeys';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { limitChanged } from 'features/gallery/store/gallerySlice';
import { debounce } from 'lodash-es';
import { memo, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold, PiWarningCircleBold } from 'react-icons/pi';
import { useListImagesQuery } from 'services/api/endpoints/images';

import GalleryImage, { GALLERY_IMAGE_CLASS_NAME } from './GalleryImage';

export const GALLERY_GRID_CLASS_NAME = 'gallery-grid';

const GalleryImageGrid = () => {
  useGalleryHotkeys();
  const { t } = useTranslation();
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

  return <Content />;
};

export default memo(GalleryImageGrid);

const Content = () => {
  const dispatch = useAppDispatch();
  const galleryImageMinimumWidth = useAppSelector((s) => s.gallery.galleryImageMinimumWidth);

  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const { imageDTOs } = useListImagesQuery(queryArgs, {
    selectFromResult: ({ data }) => ({ imageDTOs: data?.items ?? EMPTY_ARRAY }),
  });
  // Use a callback ref to get reactivity on the container element because it is conditionally rendered
  const [container, containerRef] = useState<HTMLDivElement | null>(null);

  const calculateNewLimit = useMemo(() => {
    // Debounce this to not thrash the API
    return debounce(() => {
      if (!container) {
        // Container not rendered yet
        return;
      }
      // Managing refs for dynamically rendered components is a bit tedious:
      // - https://react.dev/learn/manipulating-the-dom-with-refs#how-to-manage-a-list-of-refs-using-a-ref-callback
      // As a easy workaround, we can just grab the first gallery image element directly.
      const galleryImageEl = document.querySelector(`.${GALLERY_IMAGE_CLASS_NAME}`);
      if (!galleryImageEl) {
        // No images in gallery?
        return;
      }

      const galleryImageRect = galleryImageEl.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();

      if (!galleryImageRect.width || !galleryImageRect.height || !containerRect.width || !containerRect.height) {
        // Gallery is too small to fit images or not rendered yet
        return;
      }

      // Floating-point precision requires we round to get the correct number of images per row
      const imagesPerRow = Math.round(containerRect.width / galleryImageRect.width);
      // However, when calculating the number of images per column, we want to floor the value to not overflow the container
      const imagesPerColumn = Math.floor(containerRect.height / galleryImageRect.height);
      // Always load at least 1 row of images
      const limit = Math.max(imagesPerRow, imagesPerRow * imagesPerColumn);
      dispatch(limitChanged(limit));
    }, 300);
  }, [container, dispatch]);

  useEffect(() => {
    // We want to recalculate the limit when image size changes
    calculateNewLimit();
  }, [calculateNewLimit, galleryImageMinimumWidth]);

  useEffect(() => {
    if (!container) {
      return;
    }

    const resizeObserver = new ResizeObserver(calculateNewLimit);
    resizeObserver.observe(container);

    // First render
    calculateNewLimit();

    return () => {
      resizeObserver.disconnect();
    };
  }, [calculateNewLimit, container, dispatch]);

  return (
    <Box position="relative" w="full" h="full">
      <Box
        ref={containerRef}
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        w="full"
        h="full"
        overflow="hidden"
      >
        <Grid
          className={GALLERY_GRID_CLASS_NAME}
          gridTemplateColumns={`repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr))`}
        >
          {imageDTOs.map((imageDTO, index) => (
            <GalleryImage key={imageDTO.image_name} imageDTO={imageDTO} index={index} />
          ))}
        </Grid>
      </Box>
    </Box>
  );
};

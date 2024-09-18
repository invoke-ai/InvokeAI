import { Box, Flex, Grid } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { GallerySelectionCountTag } from 'features/gallery/components/ImageGrid/GallerySelectionCountTag';
import { useGalleryHotkeys } from 'features/gallery/hooks/useGalleryHotkeys';
import { selectGalleryImageMinimumWidth, selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { limitChanged } from 'features/gallery/store/gallerySlice';
import { debounce } from 'lodash-es';
import { memo, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold, PiWarningCircleBold } from 'react-icons/pi';
import { useListImagesQuery } from 'services/api/endpoints/images';

import { GALLERY_GRID_CLASS_NAME } from './constants';
import { GALLERY_IMAGE_CLASS_NAME, GalleryImage } from './GalleryImage';

const GalleryImageGrid = () => {
  useGalleryHotkeys();
  const { t } = useTranslation();
  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const { imageDTOs, isLoading, isError } = useListImagesQuery(queryArgs, {
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
  const galleryImageMinimumWidth = useAppSelector(selectGalleryImageMinimumWidth);

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
      const imageEl = document.querySelector(`.${GALLERY_IMAGE_CLASS_NAME}`);
      if (!imageEl) {
        // No images in gallery?
        return;
      }

      const gridEl = document.querySelector(`.${GALLERY_GRID_CLASS_NAME}`);

      if (!gridEl) {
        return;
      }

      const imageRect = imageEl.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();

      // We need to account for the gap between images
      const gridElStyle = window.getComputedStyle(gridEl);
      const gap = parseFloat(gridElStyle.gap);

      if (!imageRect.width || !imageRect.height || !containerRect.width || !containerRect.height) {
        // Gallery is too small to fit images or not rendered yet
        return;
      }

      let imagesPerColumn = 0;
      let spaceUsed = 0;

      // Floating point precision can cause imagesPerColumn to be 1 too small. Adding 1px to the container size fixes
      // this. Because the minimum image size is  without the possibility of overshooting.
      while (spaceUsed + imageRect.height <= containerRect.height + 1) {
        imagesPerColumn++; // Increment the number of images
        spaceUsed += imageRect.height; // Add image size to the used space
        if (spaceUsed + gap <= containerRect.height) {
          spaceUsed += gap; // Add gap size to the used space after each image except after the last image
        }
      }

      let imagesPerRow = 0;
      spaceUsed = 0;

      // Floating point precision can cause imagesPerRow to be 1 too small. Adding 1px to the container size fixes
      // this, without the possibility of accidentally adding an extra column.
      while (spaceUsed + imageRect.width <= containerRect.width + 1) {
        imagesPerRow++; // Increment the number of images
        spaceUsed += imageRect.width; // Add image size to the used space
        if (spaceUsed + gap <= containerRect.width) {
          spaceUsed += gap; // Add gap size to the used space after each image except after the last image
        }
      }

      // Always load at least 1 row of images
      const limit = Math.max(imagesPerRow, imagesPerRow * imagesPerColumn);

      if (queryArgs === skipToken || queryArgs.limit === limit) {
        return;
      }
      dispatch(limitChanged(limit));
    }, 300);
  }, [container, dispatch, queryArgs]);

  useEffect(() => {
    // We want to recalculate the limit when image size changes
    calculateNewLimit();
  }, [calculateNewLimit, galleryImageMinimumWidth, imageDTOs]);

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
    <Box position="relative" w="full" h="full" mt={2}>
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
          gap={1}
        >
          {imageDTOs.map((imageDTO, index) => (
            <GalleryImage key={imageDTO.image_name} imageDTO={imageDTO} index={index} />
          ))}
        </Grid>
      </Box>
      <GallerySelectionCountTag />
    </Box>
  );
};

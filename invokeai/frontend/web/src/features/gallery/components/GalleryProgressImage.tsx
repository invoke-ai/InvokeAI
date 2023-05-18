import { Flex, Image, Spinner } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { systemSelector } from 'features/system/store/systemSelectors';
import { memo } from 'react';
import { gallerySelector } from '../store/gallerySelectors';

const selector = createSelector(
  [systemSelector, gallerySelector],
  (system, gallery) => {
    const { shouldUseSingleGalleryColumn, galleryImageObjectFit } = gallery;
    const { progressImage, shouldAntialiasProgressImage } = system;

    return {
      progressImage,
      shouldUseSingleGalleryColumn,
      galleryImageObjectFit,
      shouldAntialiasProgressImage,
    };
  },
  defaultSelectorOptions
);

const GalleryProgressImage = () => {
  const {
    progressImage,
    shouldUseSingleGalleryColumn,
    galleryImageObjectFit,
    shouldAntialiasProgressImage,
  } = useAppSelector(selector);

  if (!progressImage) {
    return null;
  }

  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        aspectRatio: '1/1',
        position: 'relative',
      }}
    >
      <Image
        draggable={false}
        src={progressImage.dataURL}
        width={progressImage.width}
        height={progressImage.height}
        sx={{
          objectFit: shouldUseSingleGalleryColumn
            ? 'contain'
            : galleryImageObjectFit,
          width: '100%',
          height: '100%',
          maxWidth: '100%',
          maxHeight: '100%',
          borderRadius: 'base',
          imageRendering: shouldAntialiasProgressImage ? 'auto' : 'pixelated',
        }}
      />
      <Spinner
        sx={{ position: 'absolute', top: 1, right: 1, opacity: 0.7 }}
        speed="1.2s"
      />
    </Flex>
  );
};

export default memo(GalleryProgressImage);

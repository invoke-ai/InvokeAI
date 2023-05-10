import { Box, Flex, Image } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useGetUrl } from 'common/util/getUrl';
import { systemSelector } from 'features/system/store/systemSelectors';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash-es';

import { selectedImageSelector } from '../store/gallerySelectors';
import CurrentImageFallback from './CurrentImageFallback';
import ImageMetadataViewer from './ImageMetaDataViewer/ImageMetadataViewer';
import NextPrevImageButtons from './NextPrevImageButtons';
import CurrentImageHidden from './CurrentImageHidden';
import { memo } from 'react';

export const imagesSelector = createSelector(
  [uiSelector, selectedImageSelector, systemSelector],
  (ui, selectedImage, system) => {
    const { shouldShowImageDetails, shouldHidePreview } = ui;

    return {
      shouldShowImageDetails,
      shouldHidePreview,
      image: selectedImage,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const CurrentImagePreview = () => {
  const { shouldShowImageDetails, image, shouldHidePreview } =
    useAppSelector(imagesSelector);
  const { getUrl } = useGetUrl();

  return (
    <Flex
      sx={{
        position: 'relative',
        justifyContent: 'center',
        alignItems: 'center',
        width: '100%',
        height: '100%',
      }}
    >
      {image && (
        <Image
          src={shouldHidePreview ? undefined : getUrl(image.url)}
          width={image.metadata.width}
          height={image.metadata.height}
          fallback={shouldHidePreview ? <CurrentImageHidden /> : undefined}
          sx={{
            objectFit: 'contain',
            maxWidth: '100%',
            maxHeight: '100%',
            height: 'auto',
            position: 'absolute',
            borderRadius: 'base',
          }}
        />
      )}
      {shouldShowImageDetails && image && 'metadata' in image && (
        <Box
          sx={{
            position: 'absolute',
            top: '0',
            width: '100%',
            height: '100%',
            borderRadius: 'base',
            overflow: 'scroll',
          }}
        >
          <ImageMetadataViewer image={image} />
        </Box>
      )}
      {!shouldShowImageDetails && <NextPrevImageButtons />}
    </Flex>
  );
};

export default memo(CurrentImagePreview);

import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash-es';

import { gallerySelector } from '../store/gallerySelectors';
import CurrentImageButtons from './CurrentImageButtons';
import CurrentImagePreview from './CurrentImagePreview';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';

export const currentImageDisplaySelector = createSelector(
  [systemSelector, gallerySelector],
  (system, gallery) => {
    const { progressImage } = system;

    return {
      hasSelectedImage: Boolean(gallery.selectedImage),
      hasProgressImage: Boolean(progressImage),
    };
  },
  defaultSelectorOptions
);

/**
 * Displays the current image if there is one, plus associated actions.
 */
const CurrentImageDisplay = () => {
  const { hasSelectedImage, hasProgressImage } = useAppSelector(
    currentImageDisplaySelector
  );

  return (
    <Flex
      sx={{
        position: 'relative',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        rowGap: 4,
        borderRadius: 'base',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Flex
        flexDirection="column"
        sx={{
          w: 'full',
          h: 'full',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 4,
          position: 'absolute',
        }}
      >
        <CurrentImagePreview />
      </Flex>
      {hasSelectedImage && (
        <Box sx={{ position: 'absolute', top: 0 }}>
          <CurrentImageButtons />
        </Box>
      )}
    </Flex>
  );
};

export default CurrentImageDisplay;

import { Box, Collapse, Flex, Icon, useDisclosure } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import IAIButton from 'common/components/IAIButton';
import ImageToImageSettings from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageSettings';
import { isEqual } from 'lodash';
import { useState } from 'react';
import {
  AnimatePresence,
  motion,
  useMotionValue,
  useTransform,
} from 'framer-motion';

import { MdPhoto } from 'react-icons/md';
import {
  gallerySelector,
  selectedImageSelector,
} from '../store/gallerySelectors';
import CurrentImageButtons from './CurrentImageButtons';
import CurrentImagePreview from './CurrentImagePreview';

export const currentImageDisplaySelector = createSelector(
  [systemSelector, selectedImageSelector],
  (system, selectedImage) => {
    const { progressImage } = system;

    return {
      hasAnImageToDisplay: selectedImage || progressImage,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Displays the current image if there is one, plus associated actions.
 */
const CurrentImageDisplay = () => {
  const { hasAnImageToDisplay } = useAppSelector(currentImageDisplaySelector);
  const [shouldHideImageToImage, setShouldHideImageToImage] = useState(false);
  const w = useMotionValue(0);
  const width = useTransform(w, [0, 100], [`0px`, `100px`]);

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
        sx={{
          w: 'full',
          h: 'full',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 4,
        }}
      >
        {hasAnImageToDisplay ? (
          <CurrentImagePreview />
        ) : (
          <Icon
            as={MdPhoto}
            sx={{
              boxSize: 24,
              color: 'base.500',
            }}
          />
        )}
      </Flex>
      <Box sx={{ position: 'absolute', top: 0 }}>
        <CurrentImageButtons />
      </Box>
    </Flex>
  );
};

export default CurrentImageDisplay;

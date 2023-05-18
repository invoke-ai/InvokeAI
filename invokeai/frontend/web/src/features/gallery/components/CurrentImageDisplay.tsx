import { Flex, Icon } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash-es';

import { gallerySelector } from '../store/gallerySelectors';
import CurrentImageButtons from './CurrentImageButtons';
import CurrentImagePreview from './CurrentImagePreview';
import { FaImage } from 'react-icons/fa';

export const currentImageDisplaySelector = createSelector(
  [systemSelector, gallerySelector],
  (system, gallery) => {
    const { progressImage } = system;

    return {
      hasAnImageToDisplay: gallery.selectedImage || progressImage,
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
        }}
      >
        {hasAnImageToDisplay ? (
          <>
            <CurrentImageButtons />
            <CurrentImagePreview />
          </>
        ) : (
          <Icon
            as={FaImage}
            sx={{
              boxSize: 24,
              color: 'base.500',
            }}
          />
        )}
      </Flex>
    </Flex>
  );
};

export default CurrentImageDisplay;

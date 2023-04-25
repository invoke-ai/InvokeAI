import { Flex, Icon } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { isEqual } from 'lodash';

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

  return (
    <Flex
      sx={{
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        rowGap: 4,
        borderRadius: 'base',
      }}
    >
      {hasAnImageToDisplay ? (
        <>
          <CurrentImageButtons />
          <CurrentImagePreview />
        </>
      ) : (
        <Flex
          sx={{
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            height: '100%',
          }}
        >
          <Icon
            as={MdPhoto}
            sx={{
              boxSize: 24,
              color: 'base.500',
            }}
          />
        </Flex>
      )}
    </Flex>
  );
};

export default CurrentImageDisplay;

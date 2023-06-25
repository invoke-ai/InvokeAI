import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';

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

const CurrentImageDisplay = () => {
  const { hasSelectedImage } = useAppSelector(currentImageDisplaySelector);

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
      {hasSelectedImage && <CurrentImageButtons />}
      <CurrentImagePreview />
    </Flex>
  );
};

export default CurrentImageDisplay;

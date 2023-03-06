import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/storeHooks';
import { GalleryState } from 'features/gallery/store/gallerySlice';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash';

import { MdPhoto } from 'react-icons/md';
import { gallerySelector } from '../store/gallerySelectors';
import CurrentImageButtons from './CurrentImageButtons';
import CurrentImagePreview from './CurrentImagePreview';

export const currentImageDisplaySelector = createSelector(
  [gallerySelector, uiSelector, activeTabNameSelector],
  (gallery: GalleryState, ui, activeTabName) => {
    const { currentImage, intermediateImage } = gallery;
    const { shouldShowImageDetails } = ui;

    return {
      activeTabName,
      shouldShowImageDetails,
      hasAnImageToDisplay: currentImage || intermediateImage,
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
  const { hasAnImageToDisplay, activeTabName } = useAppSelector(
    currentImageDisplaySelector
  );

  return (
    <div className="current-image-area" data-tab-name={activeTabName}>
      {hasAnImageToDisplay ? (
        <>
          <CurrentImageButtons />
          <CurrentImagePreview />
        </>
      ) : (
        <div className="current-image-display-placeholder">
          <MdPhoto />
        </div>
      )}
    </div>
  );
};

export default CurrentImageDisplay;

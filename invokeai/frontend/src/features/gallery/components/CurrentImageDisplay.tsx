import { useAppSelector } from 'app/storeHooks';
import CurrentImageButtons from './CurrentImageButtons';
import { MdPhoto } from 'react-icons/md';
import CurrentImagePreview from './CurrentImagePreview';
import { GalleryState } from 'features/gallery/store/gallerySlice';
import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { gallerySelector } from '../store/gallerySelectors';

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
      resultEqualityCheck: _.isEqual,
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

import { RootState, useAppSelector } from 'app/store';
import CurrentImageButtons from './CurrentImageButtons';
import { MdPhoto } from 'react-icons/md';
import CurrentImagePreview from './CurrentImagePreview';
import { GalleryState } from 'features/gallery/store/gallerySlice';
import { OptionsState } from 'features/options/store/optionsSlice';
import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';

export const currentImageDisplaySelector = createSelector(
  [
    (state: RootState) => state.gallery,
    (state: RootState) => state.options,
    activeTabNameSelector,
  ],
  (gallery: GalleryState, options: OptionsState, activeTabName) => {
    const { currentImage, intermediateImage } = gallery;
    const { shouldShowImageDetails } = options;

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

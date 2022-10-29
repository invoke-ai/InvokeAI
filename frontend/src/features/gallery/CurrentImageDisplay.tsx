import { RootState, useAppSelector } from '../../app/store';
import CurrentImageButtons from './CurrentImageButtons';
import { MdPhoto } from 'react-icons/md';
import CurrentImagePreview from './CurrentImagePreview';
import { tabMap } from '../tabs/InvokeTabs';
import { GalleryState } from './gallerySlice';
import { OptionsState } from '../options/optionsSlice';
import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';

export const currentImageDisplaySelector = createSelector(
  [(state: RootState) => state.gallery, (state: RootState) => state.options],
  (gallery: GalleryState, options: OptionsState) => {
    const { currentImage, intermediateImage } = gallery;
    const { activeTab, shouldShowImageDetails } = options;

    return {
      currentImage,
      intermediateImage,
      activeTabName: tabMap[activeTab],
      shouldShowImageDetails,
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
  const {
    currentImage,
    intermediateImage,
    activeTabName,
  } = useAppSelector(currentImageDisplaySelector);

  const imageToDisplay = intermediateImage || currentImage;

  return (
    <div className="current-image-area" data-tab-name={activeTabName}>
      {imageToDisplay ? (
        <>
          <CurrentImageButtons image={imageToDisplay} />
          <CurrentImagePreview imageToDisplay={imageToDisplay} />
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

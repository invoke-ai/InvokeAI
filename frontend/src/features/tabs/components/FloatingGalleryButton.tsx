import { MdPhotoLibrary } from 'react-icons/md';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  GalleryState,
  setShouldShowGallery,
} from 'features/gallery/store/gallerySlice';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import _ from 'lodash';

const floatingGallerySelcetor = createSelector(
  [(state: RootState) => state.gallery, activeTabNameSelector],
  (gallery: GalleryState, activeTabName) => {
    const { shouldShowGallery, shouldHoldGalleryOpen, shouldPinGallery } =
      gallery;

    const shouldShowGalleryButton =
      !(shouldShowGallery || (shouldHoldGalleryOpen && !shouldPinGallery)) &&
      ['txt2img', 'img2img', 'unifiedCanvas'].includes(activeTabName);

    return {
      shouldPinGallery,
      shouldShowGalleryButton,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const FloatingGalleryButton = () => {
  const dispatch = useAppDispatch();
  const { shouldShowGalleryButton, shouldPinGallery } = useAppSelector(
    floatingGallerySelcetor
  );

  const handleShowGallery = () => {
    dispatch(setShouldShowGallery(true));
    if (shouldPinGallery) {
      dispatch(setDoesCanvasNeedScaling(true));
    }
  };

  return shouldShowGalleryButton ? (
    <IAIIconButton
      tooltip="Show Gallery (G)"
      tooltipProps={{ placement: 'top' }}
      aria-label="Show Gallery"
      styleClass="floating-show-hide-button right show-hide-button-gallery"
      onClick={handleShowGallery}
    >
      <MdPhotoLibrary />
    </IAIIconButton>
  ) : null;
};

export default FloatingGalleryButton;

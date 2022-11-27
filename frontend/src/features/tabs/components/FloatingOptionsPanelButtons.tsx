import { createSelector } from '@reduxjs/toolkit';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  OptionsState,
  setShouldShowOptionsPanel,
} from 'features/options/store/optionsSlice';
import CancelButton from 'features/options/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/options/components/ProcessButtons/InvokeButton';
import _ from 'lodash';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { FaSlidersH } from 'react-icons/fa';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import {
  GalleryState,
  setShouldShowGallery,
} from 'features/gallery/store/gallerySlice';

export const floatingSelector = createSelector(
  [
    (state: RootState) => state.options,
    (state: RootState) => state.gallery,
    activeTabNameSelector,
  ],
  (options: OptionsState, gallery: GalleryState, activeTabName) => {
    const {
      shouldPinOptionsPanel,
      shouldShowOptionsPanel,
      shouldHoldOptionsPanelOpen,
    } = options;

    const { shouldShowGallery, shouldPinGallery, shouldHoldGalleryOpen } =
      gallery;

    const shouldShowOptionsPanelButton =
      !(
        shouldShowOptionsPanel ||
        (shouldHoldOptionsPanelOpen && !shouldPinOptionsPanel)
      ) && ['txt2img', 'img2img', 'unifiedCanvas'].includes(activeTabName);

    const shouldShowGalleryButton =
      !(shouldShowGallery || (shouldHoldGalleryOpen && !shouldPinGallery)) &&
      ['txt2img', 'img2img', 'unifiedCanvas'].includes(activeTabName);

    return {
      shouldPinOptionsPanel,
      shouldShowProcessButtons:
        !shouldPinOptionsPanel || !shouldShowOptionsPanel,
      shouldShowOptionsPanelButton,
      shouldShowOptionsPanel,
      shouldShowGallery,
      shouldPinGallery,
      shouldShowGalleryButton,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const FloatingOptionsPanelButtons = () => {
  const dispatch = useAppDispatch();
  const {
    shouldShowOptionsPanel,
    shouldShowOptionsPanelButton,
    shouldShowProcessButtons,
    shouldPinOptionsPanel,
    shouldShowGallery,
    shouldPinGallery,
  } = useAppSelector(floatingSelector);

  const handleShowOptionsPanel = () => {
    dispatch(setShouldShowOptionsPanel(true));
    if (shouldPinOptionsPanel) {
      setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    }
  };

  useHotkeys(
    'f',
    () => {
      if (shouldShowGallery || shouldShowOptionsPanel) {
        dispatch(setShouldShowOptionsPanel(false));
        dispatch(setShouldShowGallery(false));
      } else {
        dispatch(setShouldShowOptionsPanel(true));
        dispatch(setShouldShowGallery(true));
      }
      if (shouldPinGallery || shouldPinOptionsPanel)
        setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    },
    [shouldShowGallery, shouldShowOptionsPanel]
  );

  return shouldShowOptionsPanelButton ? (
    <div className="show-hide-button-options">
      <IAIIconButton
        tooltip="Show Options Panel (O)"
        tooltipProps={{ placement: 'top' }}
        aria-label="Show Options Panel"
        onClick={handleShowOptionsPanel}
      >
        <FaSlidersH />
      </IAIIconButton>
      {shouldShowProcessButtons && (
        <>
          <InvokeButton iconButton />
          <CancelButton />
        </>
      )}
    </div>
  ) : null;
};

export default FloatingOptionsPanelButtons;

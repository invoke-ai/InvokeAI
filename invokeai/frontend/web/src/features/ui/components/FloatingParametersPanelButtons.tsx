import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import { GalleryState } from 'features/gallery/store/gallerySlice';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/parameters/components/ProcessButtons/InvokeButton';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { setShouldShowParametersPanel } from 'features/ui/store/uiSlice';
import { isEqual } from 'lodash';

import { FaSlidersH } from 'react-icons/fa';

export const floatingSelector = createSelector(
  [gallerySelector, uiSelector, activeTabNameSelector],
  (gallery: GalleryState, ui, activeTabName) => {
    const {
      shouldPinParametersPanel,
      shouldShowParametersPanel,
      shouldHoldParametersPanelOpen,
      shouldUseCanvasBetaLayout,
    } = ui;

    const { shouldShowGallery, shouldPinGallery, shouldHoldGalleryOpen } =
      gallery;

    const canvasBetaLayoutCheck =
      shouldUseCanvasBetaLayout && activeTabName === 'unifiedCanvas';

    const shouldShowParametersPanelButton =
      !canvasBetaLayoutCheck &&
      !(
        shouldShowParametersPanel ||
        (shouldHoldParametersPanelOpen && !shouldPinParametersPanel)
      ) &&
      ['txt2img', 'img2img', 'unifiedCanvas'].includes(activeTabName);

    const shouldShowGalleryButton =
      !(shouldShowGallery || (shouldHoldGalleryOpen && !shouldPinGallery)) &&
      ['txt2img', 'img2img', 'unifiedCanvas'].includes(activeTabName);

    const shouldShowProcessButtons =
      !canvasBetaLayoutCheck &&
      (!shouldPinParametersPanel || !shouldShowParametersPanel);

    return {
      shouldPinParametersPanel,
      shouldShowProcessButtons,
      shouldShowParametersPanelButton,
      shouldShowParametersPanel,
      shouldShowGallery,
      shouldPinGallery,
      shouldShowGalleryButton,
    };
  },
  { memoizeOptions: { resultEqualityCheck: isEqual } }
);

const FloatingParametersPanelButtons = () => {
  const dispatch = useAppDispatch();
  const {
    shouldShowParametersPanelButton,
    shouldShowProcessButtons,
    shouldPinParametersPanel,
  } = useAppSelector(floatingSelector);

  const handleShowOptionsPanel = () => {
    dispatch(setShouldShowParametersPanel(true));
    if (shouldPinParametersPanel) {
      setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    }
  };

  return shouldShowParametersPanelButton ? (
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

export default FloatingParametersPanelButtons;

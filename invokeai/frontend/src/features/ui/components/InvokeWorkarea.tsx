import { Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import ImageGallery from 'features/gallery/components/ImageGallery';
import { setInitialImage } from 'features/parameters/store/generationSlice';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { DragEvent, ReactNode } from 'react';
import { VscSplitHorizontal } from 'react-icons/vsc';

import {
  setDoesCanvasNeedScaling,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import useGetImageByUuid from 'features/gallery/hooks/useGetImageByUuid';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { setShouldShowDualDisplay } from 'features/ui/store/uiSlice';
import { isEqual } from 'lodash';

const workareaSelector = createSelector(
  [uiSelector, lightboxSelector, activeTabNameSelector],
  (ui, lightbox, activeTabName) => {
    const { shouldShowDualDisplay, shouldPinParametersPanel } = ui;
    const { isLightboxOpen } = lightbox;
    return {
      shouldShowDualDisplay,
      shouldPinParametersPanel,
      isLightboxOpen,
      shouldShowDualDisplayButton: ['inpainting'].includes(activeTabName),
      activeTabName,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type InvokeWorkareaProps = {
  optionsPanel: ReactNode;
  children: ReactNode;
  styleClass?: string;
};

const InvokeWorkarea = (props: InvokeWorkareaProps) => {
  const dispatch = useAppDispatch();
  const { optionsPanel, children, styleClass } = props;
  const {
    activeTabName,
    shouldShowDualDisplay,
    isLightboxOpen,
    shouldShowDualDisplayButton,
  } = useAppSelector(workareaSelector);

  const getImageByUuid = useGetImageByUuid();

  const handleDualDisplay = () => {
    dispatch(setShouldShowDualDisplay(!shouldShowDualDisplay));
    dispatch(setDoesCanvasNeedScaling(true));
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    const uuid = e.dataTransfer.getData('invokeai/imageUuid');
    const image = getImageByUuid(uuid);
    if (!image) return;
    if (activeTabName === 'img2img') {
      dispatch(setInitialImage(image));
    } else if (activeTabName === 'unifiedCanvas') {
      dispatch(setInitialCanvasImage(image));
    }
  };

  return (
    <div
      className={
        styleClass ? `workarea-wrapper ${styleClass}` : `workarea-wrapper`
      }
    >
      <div className="workarea-main">
        {optionsPanel}
        <div className="workarea-children-wrapper" onDrop={handleDrop}>
          {children}
          {shouldShowDualDisplayButton && (
            <Tooltip label="Toggle Split View">
              <div
                className="workarea-split-button"
                data-selected={shouldShowDualDisplay}
                onClick={handleDualDisplay}
              >
                <VscSplitHorizontal />
              </div>
            </Tooltip>
          )}
        </div>
        {!isLightboxOpen && <ImageGallery />}
      </div>
    </div>
  );
};

export default InvokeWorkarea;

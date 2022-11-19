import { Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { ReactNode } from 'react';
import { VscSplitHorizontal } from 'react-icons/vsc';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import ImageGallery from 'features/gallery/components/ImageGallery';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import {
  OptionsState,
  setShowDualDisplay,
} from 'features/options/store/optionsSlice';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';

const workareaSelector = createSelector(
  [(state: RootState) => state.options, activeTabNameSelector],
  (options: OptionsState, activeTabName) => {
    const { showDualDisplay, shouldPinOptionsPanel, isLightBoxOpen } = options;
    return {
      showDualDisplay,
      shouldPinOptionsPanel,
      isLightBoxOpen,
      shouldShowDualDisplayButton: ['inpainting'].includes(activeTabName),
    };
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
  const { showDualDisplay, isLightBoxOpen, shouldShowDualDisplayButton } =
    useAppSelector(workareaSelector);

  const handleDualDisplay = () => {
    dispatch(setShowDualDisplay(!showDualDisplay));
    dispatch(setDoesCanvasNeedScaling(true));
  };

  return (
    <div
      className={
        styleClass ? `workarea-wrapper ${styleClass}` : `workarea-wrapper`
      }
    >
      <div className="workarea-main">
        {optionsPanel}
        <div className="workarea-children-wrapper">
          {children}
          {shouldShowDualDisplayButton && (
            <Tooltip label="Toggle Split View">
              <div
                className="workarea-split-button"
                data-selected={showDualDisplay}
                onClick={handleDualDisplay}
              >
                <VscSplitHorizontal />
              </div>
            </Tooltip>
          )}
        </div>
        {!isLightBoxOpen && <ImageGallery />}
      </div>
    </div>
  );
};

export default InvokeWorkarea;

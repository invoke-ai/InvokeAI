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

const floatingOptionsSelector = createSelector(
  [(state: RootState) => state.options, activeTabNameSelector],
  (options: OptionsState, activeTabName) => {
    const {
      shouldPinOptionsPanel,
      shouldShowOptionsPanel,
      shouldHoldOptionsPanelOpen,
    } = options;

    const shouldShowOptionsPanelButton =
      !(
        shouldShowOptionsPanel ||
        (shouldHoldOptionsPanelOpen && !shouldPinOptionsPanel)
      ) && ['txt2img', 'img2img', 'unifiedCanvas'].includes(activeTabName);

    return {
      shouldPinOptionsPanel,
      shouldShowProcessButtons:
        !shouldPinOptionsPanel || !shouldShowOptionsPanel,
      shouldShowOptionsPanelButton,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const FloatingOptionsPanelButtons = () => {
  const dispatch = useAppDispatch();
  const {
    shouldShowOptionsPanelButton,
    shouldShowProcessButtons,
    shouldPinOptionsPanel,
  } = useAppSelector(floatingOptionsSelector);

  const handleShowOptionsPanel = () => {
    dispatch(setShouldShowOptionsPanel(true));
    if (shouldPinOptionsPanel) {
      setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    }
  };

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

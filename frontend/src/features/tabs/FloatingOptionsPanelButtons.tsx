import { createSelector } from '@reduxjs/toolkit';
import { IoMdOptions } from 'react-icons/io';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  OptionsState,
  setShouldShowOptionsPanel,
} from 'features/options/optionsSlice';
import CancelButton from 'features/options/ProcessButtons/CancelButton';
import InvokeButton from 'features/options/ProcessButtons/InvokeButton';
import _ from 'lodash';
import LoopbackButton from 'features/options/ProcessButtons/Loopback';
import { setDoesCanvasNeedScaling } from 'features/canvas/canvasSlice';

const canInvokeSelector = createSelector(
  (state: RootState) => state.options,

  (options: OptionsState) => {
    const { shouldPinOptionsPanel, shouldShowOptionsPanel } = options;
    return {
      shouldPinOptionsPanel,
      shouldShowProcessButtons:
        !shouldPinOptionsPanel || !shouldShowOptionsPanel,
    };
  },
  { memoizeOptions: { resultEqualityCheck: _.isEqual } }
);

const FloatingOptionsPanelButtons = () => {
  const dispatch = useAppDispatch();
  const { shouldShowProcessButtons, shouldPinOptionsPanel } =
    useAppSelector(canInvokeSelector);

  const handleShowOptionsPanel = () => {
    dispatch(setShouldShowOptionsPanel(true));
    if (shouldPinOptionsPanel) {
      setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    }
  };

  return (
    <div className="show-hide-button-options">
      <IAIIconButton
        tooltip="Show Options Panel (O)"
        tooltipProps={{ placement: 'top' }}
        aria-label="Show Options Panel"
        onClick={handleShowOptionsPanel}
      >
        <IoMdOptions />
      </IAIIconButton>
      {shouldShowProcessButtons && (
        <>
          <InvokeButton iconButton />
          <LoopbackButton />
          <CancelButton />
        </>
      )}
    </div>
  );
};

export default FloatingOptionsPanelButtons;

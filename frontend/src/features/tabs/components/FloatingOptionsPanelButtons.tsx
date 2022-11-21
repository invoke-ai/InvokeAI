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
        <FaSlidersH />
      </IAIIconButton>
      {shouldShowProcessButtons && (
        <>
          <InvokeButton iconButton />
          <CancelButton />
        </>
      )}
    </div>
  );
};

export default FloatingOptionsPanelButtons;

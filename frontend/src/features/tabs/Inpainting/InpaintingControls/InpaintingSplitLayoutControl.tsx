import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { VscSplitHorizontal } from 'react-icons/vsc';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAIIconButton from '../../../../common/components/IAIIconButton';
import { setShowDualDisplay } from '../../../options/optionsSlice';
import { setNeedsCache } from '../inpaintingSlice';

export default function InpaintingSplitLayoutControl() {
  const dispatch = useAppDispatch();
  const showDualDisplay = useAppSelector(
    (state: RootState) => state.options.showDualDisplay
  );

  const handleDualDisplay = () => {
    dispatch(setShowDualDisplay(!showDualDisplay));
    dispatch(setNeedsCache(true));
  };

  // Hotkeys
  // Toggle split view
  useHotkeys(
    'shift+j',
    () => {
      handleDualDisplay();
    },
    [showDualDisplay]
  );

  return (
    <IAIIconButton
      aria-label="Split Layout (Shift+J)"
      tooltip="Split Layout (Shift+J)"
      icon={<VscSplitHorizontal />}
      data-selected={showDualDisplay}
      onClick={handleDualDisplay}
    />
  );
}

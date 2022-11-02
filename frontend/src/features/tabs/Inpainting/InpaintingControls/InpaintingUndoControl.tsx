import { createSelector } from '@reduxjs/toolkit';
import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaUndo } from 'react-icons/fa';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAIIconButton from '../../../../common/components/IAIIconButton';
import { InpaintingState, undo } from '../inpaintingSlice';

import _ from 'lodash';
import { activeTabNameSelector } from '../../../options/optionsSelectors';

const inpaintingUndoSelector = createSelector(
  [(state: RootState) => state.inpainting, activeTabNameSelector],
  (inpainting: InpaintingState, activeTabName) => {
    const { pastLines, shouldShowMask } = inpainting;

    return {
      canUndo: pastLines.length > 0,
      shouldShowMask,
      activeTabName,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function InpaintingUndoControl() {
  const dispatch = useAppDispatch();

  const { canUndo, shouldShowMask, activeTabName } = useAppSelector(
    inpaintingUndoSelector
  );

  const handleUndo = () => dispatch(undo());

  // Hotkeys
  // Undo
  useHotkeys(
    'cmd+z, control+z',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleUndo();
    },
    {
      enabled: activeTabName === 'inpainting' && shouldShowMask && canUndo,
    },
    [activeTabName, shouldShowMask, canUndo]
  );

  return (
    <IAIIconButton
      aria-label="Undo"
      tooltip="Undo"
      icon={<FaUndo />}
      onClick={handleUndo}
      isDisabled={!canUndo || !shouldShowMask}
    />
  );
}

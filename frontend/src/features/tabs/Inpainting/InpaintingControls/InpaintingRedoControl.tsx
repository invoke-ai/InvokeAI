import { createSelector } from '@reduxjs/toolkit';
import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaRedo } from 'react-icons/fa';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAIIconButton from '../../../../common/components/IAIIconButton';
import { activeTabNameSelector } from '../../../options/optionsSelectors';
import { InpaintingState, redo } from '../inpaintingSlice';

import _ from 'lodash';

const inpaintingRedoSelector = createSelector(
  [(state: RootState) => state.inpainting, activeTabNameSelector],
  (inpainting: InpaintingState, activeTabName) => {
    const { futureLines, shouldShowMask } = inpainting;

    return {
      canRedo: futureLines.length > 0,
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

export default function InpaintingRedoControl() {
  const dispatch = useAppDispatch();
  const { canRedo, shouldShowMask, activeTabName } = useAppSelector(
    inpaintingRedoSelector
  );

  const handleRedo = () => dispatch(redo());

  // Hotkeys

  // Redo
  useHotkeys(
    'cmd+shift+z, control+shift+z, control+y, cmd+y',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleRedo();
    },
    {
      enabled: activeTabName === 'inpainting' && shouldShowMask && canRedo,
    },
    [activeTabName, shouldShowMask, canRedo]
  );

  return (
    <IAIIconButton
      aria-label="Redo"
      tooltip="Redo"
      icon={<FaRedo />}
      onClick={handleRedo}
      isDisabled={!canRedo || !shouldShowMask}
    />
  );
}

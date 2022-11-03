import { createSelector } from '@reduxjs/toolkit';
import React from 'react';
import { MdInvertColors, MdInvertColorsOff } from 'react-icons/md';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../../app/store';
import IAIIconButton from '../../../../../common/components/IAIIconButton';
import { InpaintingState, setShouldInvertMask } from '../../inpaintingSlice';

import _ from 'lodash';
import { activeTabNameSelector } from '../../../../options/optionsSelectors';
import { useHotkeys } from 'react-hotkeys-hook';

const inpaintingMaskInvertSelector = createSelector(
  [(state: RootState) => state.inpainting, activeTabNameSelector],
  (inpainting: InpaintingState, activeTabName) => {
    const { shouldShowMask, shouldInvertMask } = inpainting;

    return { shouldInvertMask, shouldShowMask, activeTabName };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function InpaintingMaskInvertControl() {
  const { shouldInvertMask, shouldShowMask, activeTabName } = useAppSelector(
    inpaintingMaskInvertSelector
  );
  const dispatch = useAppDispatch();

  const handleToggleShouldInvertMask = () =>
    dispatch(setShouldInvertMask(!shouldInvertMask));

  // Invert mask
  useHotkeys(
    'shift+m',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleToggleShouldInvertMask();
    },
    {
      enabled: activeTabName === 'inpainting' && shouldShowMask,
    },
    [activeTabName, shouldInvertMask, shouldShowMask]
  );

  return (
    <IAIIconButton
      tooltip="Invert Mask Display (Shift+M)"
      aria-label="Invert Mask Display (Shift+M)"
      data-selected={shouldInvertMask}
      icon={
        shouldInvertMask ? (
          <MdInvertColors size={22} />
        ) : (
          <MdInvertColorsOff size={22} />
        )
      }
      onClick={handleToggleShouldInvertMask}
      isDisabled={!shouldShowMask}
    />
  );
}

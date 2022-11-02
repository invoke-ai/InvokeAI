import React from 'react';
import { RgbaColor } from 'react-colorful';
import { FaPalette } from 'react-icons/fa';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../../app/store';
import IAIColorPicker from '../../../../../common/components/IAIColorPicker';
import IAIIconButton from '../../../../../common/components/IAIIconButton';
import IAIPopover from '../../../../../common/components/IAIPopover';
import { InpaintingState, setMaskColor } from '../../inpaintingSlice';

import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector } from '../../../../options/optionsSelectors';
import { useHotkeys } from 'react-hotkeys-hook';

const inpaintingMaskColorPickerSelector = createSelector(
  [(state: RootState) => state.inpainting, activeTabNameSelector],
  (inpainting: InpaintingState, activeTabName) => {
    const { shouldShowMask, maskColor } = inpainting;

    return { shouldShowMask, maskColor, activeTabName };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function InpaintingMaskColorPicker() {
  const { shouldShowMask, maskColor, activeTabName } = useAppSelector(
    inpaintingMaskColorPickerSelector
  );
  const dispatch = useAppDispatch();
  const handleChangeMaskColor = (newColor: RgbaColor) => {
    dispatch(setMaskColor(newColor));
  };

  // Hotkeys
  // Decrease mask opacity
  useHotkeys(
    'shift+[',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleChangeMaskColor({
        ...maskColor,
        a: Math.max(maskColor.a - 0.05, 0),
      });
    },
    {
      enabled: activeTabName === 'inpainting' && shouldShowMask,
    },
    [activeTabName, shouldShowMask, maskColor.a]
  );

  // Increase mask opacity
  useHotkeys(
    'shift+]',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleChangeMaskColor({
        ...maskColor,
        a: Math.min(maskColor.a + 0.05, 100),
      });
    },
    {
      enabled: activeTabName === 'inpainting' && shouldShowMask,
    },
    [activeTabName, shouldShowMask, maskColor.a]
  );

  return (
    <IAIPopover
      trigger="hover"
      styleClass="inpainting-color-picker"
      triggerComponent={
        <IAIIconButton
          aria-label="Mask Color"
          icon={<FaPalette />}
          isDisabled={!shouldShowMask}
          cursor={'pointer'}
        />
      }
    >
      <IAIColorPicker color={maskColor} onChange={handleChangeMaskColor} />
    </IAIPopover>
  );
}

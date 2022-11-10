import { Button, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  clearMask,
  currentCanvasSelector,
  setIsMaskEnabled,
  setLayer,
  setMaskColor,
} from './canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaMask } from 'react-icons/fa';
import IAIPopover from 'common/components/IAIPopover';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIColorPicker from 'common/components/IAIColorPicker';

export const selector = createSelector(
  [currentCanvasSelector],
  (currentCanvas) => {
    const { maskColor, layer, isMaskEnabled } = currentCanvas;

    return {
      layer,
      maskColor,
      isMaskEnabled,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);
const IAICanvasMaskButtonPopover = () => {
  const dispatch = useAppDispatch();
  const { layer, maskColor, isMaskEnabled } = useAppSelector(selector);

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          aria-label="Select Mask Layer"
          tooltip="Select Mask Layer"
          data-alert={layer === 'mask'}
          onClick={() => dispatch(setLayer(layer === 'mask' ? 'base' : 'mask'))}
          icon={<FaMask />}
        />
      }
    >
      <Flex direction={'column'} gap={'0.5rem'}>
        <Button onClick={() => dispatch(clearMask())}>Clear Mask</Button>
        <IAICheckbox
          label="Enable Mask"
          isChecked={isMaskEnabled}
          onChange={(e) => dispatch(setIsMaskEnabled(e.target.checked))}
        />
        <IAICheckbox label="Invert Mask" />
        <IAIColorPicker
          color={maskColor}
          onChange={(newColor) => dispatch(setMaskColor(newColor))}
        />
      </Flex>
    </IAIPopover>
  );
};

export default IAICanvasMaskButtonPopover;

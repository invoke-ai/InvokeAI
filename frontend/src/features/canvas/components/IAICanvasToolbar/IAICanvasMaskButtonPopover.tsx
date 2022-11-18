import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  clearMask,
  setIsMaskEnabled,
  setLayer,
  setMaskColor,
  setShouldPreserveMaskedArea,
} from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaMask } from 'react-icons/fa';
import IAIPopover from 'common/components/IAIPopover';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIColorPicker from 'common/components/IAIColorPicker';
import IAIButton from 'common/components/IAIButton';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { useHotkeys } from 'react-hotkeys-hook';

export const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { maskColor, layer, isMaskEnabled, shouldPreserveMaskedArea } =
      canvas;

    return {
      layer,
      maskColor,
      isMaskEnabled,
      shouldPreserveMaskedArea,
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
  const { layer, maskColor, isMaskEnabled, shouldPreserveMaskedArea } =
    useAppSelector(selector);

  useHotkeys(
    ['q'],
    () => {
      handleToggleMaskLayer();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [layer]
  );

  useHotkeys(
    ['shift+c'],
    () => {
      handleClearMask();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    []
  );

  useHotkeys(
    ['h'],
    () => {
      handleToggleEnableMask();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [isMaskEnabled]
  );

  const handleToggleMaskLayer = () => {
    dispatch(setLayer(layer === 'mask' ? 'base' : 'mask'));
  };

  const handleClearMask = () => dispatch(clearMask());

  const handleToggleEnableMask = () =>
    dispatch(setIsMaskEnabled(!isMaskEnabled));

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          aria-label="Select Mask Layer (Q)"
          tooltip="Select Mask Layer (Q)"
          data-alert={layer === 'mask'}
          onClick={handleToggleMaskLayer}
          icon={<FaMask />}
        />
      }
    >
      <Flex direction={'column'} gap={'0.5rem'}>
        <IAIButton onClick={handleClearMask} tooltip={'Clear Mask (Shift+C)'}>
          Clear Mask
        </IAIButton>
        <IAICheckbox
          label="Enable Mask (H)"
          isChecked={isMaskEnabled}
          onChange={handleToggleEnableMask}
        />
        <IAICheckbox
          label="Preserve Masked Area"
          isChecked={shouldPreserveMaskedArea}
          onChange={(e) =>
            dispatch(setShouldPreserveMaskedArea(e.target.checked))
          }
        />
        <IAIColorPicker
          color={maskColor}
          onChange={(newColor) => dispatch(setMaskColor(newColor))}
        />
      </Flex>
    </IAIPopover>
  );
};

export default IAICanvasMaskButtonPopover;

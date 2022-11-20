import { Box, ButtonGroup, Flex } from '@chakra-ui/react';
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
import { FaMask, FaTrash } from 'react-icons/fa';
import IAIPopover from 'common/components/IAIPopover';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIColorPicker from 'common/components/IAIColorPicker';
import IAIButton from 'common/components/IAIButton';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import IAISelect from 'common/components/IAISelect';
import {
  CanvasLayer,
  LAYER_NAMES_DICT,
} from 'features/canvas/store/canvasTypes';
import { ChangeEvent } from 'react';
import {
  rgbaColorToRgbString,
  rgbaColorToString,
} from 'features/canvas/util/colorToString';

export const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { maskColor, layer, isMaskEnabled, shouldPreserveMaskedArea } =
      canvas;

    return {
      layer,
      maskColor,
      maskColorString: rgbaColorToString(maskColor),
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
const IAICanvasMaskOptions = () => {
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
    <>
      <IAISelect
        label={'Layer'}
        tooltipProps={{ hasArrow: true, placement: 'top' }}
        value={isMaskEnabled ? layer : 'base'}
        isDisabled={!isMaskEnabled}
        validValues={LAYER_NAMES_DICT}
        onChange={(e: ChangeEvent<HTMLSelectElement>) =>
          dispatch(setLayer(e.target.value as CanvasLayer))
        }
      />
      <IAIPopover
        trigger="hover"
        triggerComponent={
          <ButtonGroup>
            <IAIIconButton
              aria-label="Inpainting Mask Options"
              tooltip="Inpainting Mask Options"
              data-alert={layer === 'mask'}
              icon={<FaMask />}
            />
          </ButtonGroup>
        }
      >
        <Flex direction={'column'} gap={'0.5rem'}>
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
            style={{ paddingTop: '0.5rem', paddingBottom: '0.5rem' }}
            color={maskColor}
            onChange={(newColor) => dispatch(setMaskColor(newColor))}
          />
          <IAIButton
            size={'sm'}
            leftIcon={<FaTrash />}
            onClick={handleClearMask}
          >
            Clear Mask (Shift+C)
          </IAIButton>
        </Flex>
      </IAIPopover>
    </>
  );
};

export default IAICanvasMaskOptions;

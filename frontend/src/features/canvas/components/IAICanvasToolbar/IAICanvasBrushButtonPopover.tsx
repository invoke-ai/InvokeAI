import { createSelector } from '@reduxjs/toolkit';
import {
  setBrushColor,
  setBrushSize,
  setTool,
} from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaPaintBrush } from 'react-icons/fa';
import IAIPopover from 'common/components/IAIPopover';
import IAIColorPicker from 'common/components/IAIColorPicker';
import IAISlider from 'common/components/IAISlider';
import { Flex } from '@chakra-ui/react';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { useHotkeys } from 'react-hotkeys-hook';

export const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const { brushColor, brushSize, tool } = canvas;

    return {
      tool,
      brushColor,
      brushSize,
      isStaging,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const IAICanvasBrushButtonPopover = () => {
  const dispatch = useAppDispatch();
  const { tool, brushColor, brushSize, isStaging } = useAppSelector(selector);

  useHotkeys(
    ['b'],
    () => {
      handleSelectBrushTool();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    []
  );

  useHotkeys(
    ['['],
    () => {
      dispatch(setBrushSize(Math.max(brushSize - 5, 5)));
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [brushSize]
  );

  useHotkeys(
    [']'],
    () => {
      dispatch(setBrushSize(Math.min(brushSize + 5, 500)));
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [brushSize]
  );

  const handleSelectBrushTool = () => dispatch(setTool('brush'));

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          aria-label="Brush (B)"
          tooltip="Brush (B)"
          icon={<FaPaintBrush />}
          data-selected={tool === 'brush' && !isStaging}
          onClick={handleSelectBrushTool}
          isDisabled={isStaging}
        />
      }
    >
      <Flex minWidth={'15rem'} direction={'column'} gap={'1rem'} width={'100%'}>
        <Flex gap={'1rem'} justifyContent="space-between">
          <IAISlider
            label="Size"
            value={brushSize}
            withInput
            onChange={(newSize) => dispatch(setBrushSize(newSize))}
          />
        </Flex>
        <IAIColorPicker
          style={{ width: '100%' }}
          color={brushColor}
          onChange={(newColor) => dispatch(setBrushColor(newColor))}
        />
      </Flex>
    </IAIPopover>
  );
};

export default IAICanvasBrushButtonPopover;

import { createSelector } from '@reduxjs/toolkit';
import { setBrushSize, setTool } from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaEraser } from 'react-icons/fa';
import IAIPopover from 'common/components/IAIPopover';
import IAISlider from 'common/components/IAISlider';
import { Flex } from '@chakra-ui/react';
import { useHotkeys } from 'react-hotkeys-hook';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';

export const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const { brushSize, tool } = canvas;

    return {
      tool,
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
const IAICanvasEraserButtonPopover = () => {
  const dispatch = useAppDispatch();
  const { tool, brushSize, isStaging } = useAppSelector(selector);

  const handleSelectEraserTool = () => dispatch(setTool('eraser'));

  useHotkeys(
    ['e'],
    () => {
      handleSelectEraserTool();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [tool]
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

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          aria-label="Eraser Tool (E)"
          tooltip="Eraser Tool (E)"
          icon={<FaEraser />}
          data-selected={tool === 'eraser' && !isStaging}
          isDisabled={isStaging}
          onClick={() => dispatch(setTool('eraser'))}
        />
      }
    >
      <Flex minWidth={'15rem'} direction={'column'} gap={'1rem'}>
        <IAISlider
          label="Size"
          value={brushSize}
          withInput
          onChange={(newSize) => dispatch(setBrushSize(newSize))}
        />
      </Flex>
    </IAIPopover>
  );
};

export default IAICanvasEraserButtonPopover;

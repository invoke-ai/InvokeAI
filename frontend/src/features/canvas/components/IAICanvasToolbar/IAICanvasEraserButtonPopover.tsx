import { createSelector } from '@reduxjs/toolkit';
import { setEraserSize, setTool } from 'features/canvas/store/canvasSlice';
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
    const { eraserSize, tool } = canvas;

    return {
      tool,
      eraserSize,
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
  const { tool, eraserSize, isStaging } = useAppSelector(selector);

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
      dispatch(setEraserSize(Math.max(eraserSize - 5, 5)));
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [eraserSize]
  );

  useHotkeys(
    [']'],
    () => {
      dispatch(setEraserSize(Math.min(eraserSize + 5, 500)));
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [eraserSize]
  );

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          aria-label="Eraser (E)"
          tooltip="Eraser (E)"
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
          value={eraserSize}
          withInput
          onChange={(newSize) => dispatch(setEraserSize(newSize))}
        />
      </Flex>
    </IAIPopover>
  );
};

export default IAICanvasEraserButtonPopover;

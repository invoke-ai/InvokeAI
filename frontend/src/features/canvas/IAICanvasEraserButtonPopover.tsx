import { createSelector } from '@reduxjs/toolkit';
import { currentCanvasSelector, setEraserSize, setTool } from './canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaEraser } from 'react-icons/fa';
import IAIPopover from 'common/components/IAIPopover';
import IAISlider from 'common/components/IAISlider';
import { Flex } from '@chakra-ui/react';
import { useHotkeys } from 'react-hotkeys-hook';

export const selector = createSelector(
  [currentCanvasSelector],
  (currentCanvas) => {
    const { eraserSize, tool } = currentCanvas;

    return {
      tool,
      eraserSize,
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
  const { tool, eraserSize } = useAppSelector(selector);

  const handleSelectEraserTool = () => dispatch(setTool('eraser'));

  useHotkeys(
    'e',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleSelectEraserTool();
    },
    {
      enabled: true,
    },
    [tool]
  );

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          aria-label="Eraser (E)"
          tooltip="Eraser (E)"
          icon={<FaEraser />}
          data-selected={tool === 'eraser'}
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

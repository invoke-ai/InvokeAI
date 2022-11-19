import { ButtonGroup, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  resetCanvas,
  resetCanvasView,
  resizeAndScaleCanvas,
  setBrushColor,
  setBrushSize,
  setTool,
} from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/store';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  FaArrowsAlt,
  FaEraser,
  FaPaintBrush,
  FaSlidersH,
} from 'react-icons/fa';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import IAICanvasBrushButtonPopover from './IAICanvasBrushButtonPopover';
import IAICanvasEraserButtonPopover from './IAICanvasEraserButtonPopover';
import { useHotkeys } from 'react-hotkeys-hook';
import IAIPopover from 'common/components/IAIPopover';
import IAISlider from 'common/components/IAISlider';
import IAIColorPicker from 'common/components/IAIColorPicker';
import { rgbaColorToString } from 'features/canvas/util/colorToString';

export const selector = createSelector(
  [canvasSelector, isStagingSelector, systemSelector],
  (canvas, isStaging, system) => {
    const { isProcessing } = system;
    const { tool, brushColor, brushSize } = canvas;

    return {
      tool,
      isStaging,
      isProcessing,
      brushColor,
      brushColorString: rgbaColorToString(brushColor),
      brushSize,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const IAICanvasToolChooserOptions = () => {
  const dispatch = useAppDispatch();
  const { tool, brushColor, brushSize, brushColorString, isStaging } =
    useAppSelector(selector);

  useHotkeys(
    ['v'],
    () => {
      handleSelectMoveTool();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    []
  );

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

  const handleSelectBrushTool = () => dispatch(setTool('brush'));
  const handleSelectEraserTool = () => dispatch(setTool('eraser'));
  const handleSelectMoveTool = () => dispatch(setTool('move'));

  return (
    <ButtonGroup isAttached>
      <IAIIconButton
        aria-label="Brush Tool (B)"
        tooltip="Brush Tool (B)"
        icon={<FaPaintBrush />}
        data-selected={tool === 'brush' && !isStaging}
        onClick={handleSelectBrushTool}
        isDisabled={isStaging}
      />
      <IAIIconButton
        aria-label="Eraser Tool (E)"
        tooltip="Eraser Tool (E)"
        icon={<FaEraser />}
        data-selected={tool === 'eraser' && !isStaging}
        isDisabled={isStaging}
        onClick={() => dispatch(setTool('eraser'))}
      />
      <IAIIconButton
        aria-label="Move Tool (V)"
        tooltip="Move Tool (V)"
        icon={<FaArrowsAlt />}
        data-selected={tool === 'move' || isStaging}
        onClick={handleSelectMoveTool}
      />

      <IAIPopover
        trigger="hover"
        triggerComponent={
          <IAIIconButton
            aria-label="Tool Options"
            tooltip="Tool Options"
            icon={<FaSlidersH />}
          />
        }
      >
        <Flex
          minWidth={'15rem'}
          direction={'column'}
          gap={'1rem'}
          width={'100%'}
        >
          <Flex gap={'1rem'} justifyContent="space-between">
            <IAISlider
              label="Size"
              value={brushSize}
              withInput
              onChange={(newSize) => dispatch(setBrushSize(newSize))}
              sliderNumberInputProps={{ max: 500 }}
              inputReadOnly={false}
            />
          </Flex>
          <IAIColorPicker
            style={{
              width: '100%',
              paddingTop: '0.5rem',
              paddingBottom: '0.5rem',
            }}
            color={brushColor}
            onChange={(newColor) => dispatch(setBrushColor(newColor))}
          />
        </Flex>
      </IAIPopover>
    </ButtonGroup>
  );
};

export default IAICanvasToolChooserOptions;

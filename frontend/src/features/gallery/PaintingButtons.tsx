import {
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Switch,
  Flex,
  FormControl,
  FormLabel,
  SliderMark,
  useToast,
} from '@chakra-ui/react';

import { useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaEraser, FaPaintBrush, FaTrash } from 'react-icons/fa';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import IAIIconButton from '../../common/components/IAIIconButton';
import PaintingCanvas from './Canvas/PaintingCanvas';
import {
  OptionsState,
  setInpaintingBrushSize,
  setInpaintingTool,
} from '../options/optionsSlice';
import { createSelector } from '@reduxjs/toolkit';
import { tabMap } from '../tabs/InvokeTabs';
import { isEqual } from 'lodash';

import { canvasRef } from './Canvas/PaintingCanvas';

export const inpaintingOptionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      tool: options.inpaintingTool,
      brushSize: options.inpaintingBrushSize,
      brushShape: options.inpaintingBrushShape,
      // this seems to be a reasonable calculation to get a good brush stamp pixel distance
      brushIncrement: Math.floor(
        Math.min(Math.max(options.inpaintingBrushSize / 8, 1), 5)
      ),
      activeTab: tabMap[options.activeTab],
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const PaintingButtons = () => {
  const { tool, brushSize, activeTab } = useAppSelector(
    inpaintingOptionsSelector
  );

  const dispatch = useAppDispatch();
  const toast = useToast();

  // TODO: add mask invert display (so u can see exactly what parts of image are masked)
  const [shouldInvertMask, setShouldInvertMask] = useState<boolean>(false);

  // TODO: add mask overlay display
  const [shouldOverlayMask, setShouldOverlayMask] = useState<boolean>(false);

  // Hotkeys
  useHotkeys(
    '[',
    () => {
      if (activeTab === 'inpainting' && brushSize - 5 > 0) {
        dispatch(setInpaintingBrushSize(brushSize - 5));
      } else {
        dispatch(setInpaintingBrushSize(1));
      }
    },
    [brushSize]
  );

  useHotkeys(
    ']',
    () => {
      if (activeTab === 'inpainting') {
        dispatch(setInpaintingBrushSize(brushSize + 5));
      }
    },
    [brushSize]
  );

  useHotkeys('b', () => {
    if (activeTab === 'inpainting') {
      dispatch(setInpaintingTool('eraser'));
    }
  });

  useHotkeys('e', () => {
    if (activeTab === 'inpainting') {
      dispatch(setInpaintingTool('uneraser'));
    }
  });

  return (
    <Flex gap={4} padding={2}>
      <Flex gap={4}>
        <IAIIconButton
          aria-label="Eraser"
          tooltip="Eraser"
          icon={<FaEraser />}
          colorScheme={tool === 'eraser' ? 'green' : undefined}
          onClick={() => dispatch(setInpaintingTool('eraser'))}
        />
        <IAIIconButton
          aria-label="Un-eraser"
          tooltip="Un-eraser"
          icon={<FaPaintBrush />}
          colorScheme={tool === 'uneraser' ? 'green' : undefined}
          onClick={() => dispatch(setInpaintingTool('uneraser'))}
        />
        <IAIIconButton
          aria-label="Clear mask"
          tooltip="Clear mask"
          icon={<FaTrash />}
          colorScheme={'red'}
        />
      </Flex>
      <Flex gap={4}>
        <FormControl width={300}>
          <FormLabel>Brush Radius</FormLabel>

          <Slider
            aria-label="radius"
            value={brushSize}
            onChange={(v: number) => {
              dispatch(setInpaintingBrushSize(v));
            }}
            min={1}
            max={500}
          >
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <SliderMark
              value={brushSize}
              textAlign="center"
              bg="gray.800"
              color="white"
              mt="-10"
              ml="-5"
              w="12"
            >
              {brushSize}px
            </SliderMark>
            <SliderThumb />
          </Slider>
        </FormControl>
        <FormControl width={300}>
          <FormLabel>Invert Mask Display</FormLabel>

          <Switch
            isDisabled={true}
            checked={shouldInvertMask}
            onChange={(e) => setShouldInvertMask(e.target.checked)}
          />
        </FormControl>
      </Flex>
    </Flex>
  )
}

export default PaintingButtons;
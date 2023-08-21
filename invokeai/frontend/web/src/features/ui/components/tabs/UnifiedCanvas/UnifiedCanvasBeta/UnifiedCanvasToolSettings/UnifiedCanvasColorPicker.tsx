import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import IAIPopover from 'common/components/IAIPopover';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { setBrushColor, setMaskColor } from 'features/canvas/store/canvasSlice';
import { clamp, isEqual } from 'lodash-es';

import { useHotkeys } from 'react-hotkeys-hook';

const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const { brushColor, maskColor, layer } = canvas;
    return {
      brushColor,
      maskColor,
      layer,
      isStaging,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export default function UnifiedCanvasColorPicker() {
  const dispatch = useAppDispatch();
  const { brushColor, maskColor, layer, isStaging } = useAppSelector(selector);

  const currentColorDisplay = () => {
    if (layer === 'base') {
      return `rgba(${brushColor.r},${brushColor.g},${brushColor.b},${brushColor.a})`;
    }
    if (layer === 'mask') {
      return `rgba(${maskColor.r},${maskColor.g},${maskColor.b},${maskColor.a})`;
    }
  };

  useHotkeys(
    ['shift+BracketLeft'],
    () => {
      dispatch(
        setBrushColor({
          ...brushColor,
          a: clamp(brushColor.a - 0.05, 0.05, 1),
        })
      );
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushColor]
  );

  useHotkeys(
    ['shift+BracketRight'],
    () => {
      dispatch(
        setBrushColor({
          ...brushColor,
          a: clamp(brushColor.a + 0.05, 0.05, 1),
        })
      );
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushColor]
  );

  return (
    <IAIPopover
      triggerComponent={
        <Box
          sx={{
            width: 7,
            height: 7,
            minWidth: 7,
            minHeight: 7,
            borderRadius: 'full',
            bg: currentColorDisplay(),
            cursor: 'pointer',
          }}
        />
      }
    >
      <Flex minWidth={60} direction="column" gap={4} width="100%">
        {layer === 'base' && (
          <IAIColorPicker
            sx={{
              width: '100%',
              paddingTop: 2,
              paddingBottom: 2,
            }}
            pickerColor={brushColor}
            onChange={(newColor) => dispatch(setBrushColor(newColor))}
          />
        )}
        {layer === 'mask' && (
          <IAIColorPicker
            sx={{
              width: '100%',
              paddingTop: 2,
              paddingBottom: 2,
            }}
            pickerColor={maskColor}
            onChange={(newColor) => dispatch(setMaskColor(newColor))}
          />
        )}
      </Flex>
    </IAIPopover>
  );
}

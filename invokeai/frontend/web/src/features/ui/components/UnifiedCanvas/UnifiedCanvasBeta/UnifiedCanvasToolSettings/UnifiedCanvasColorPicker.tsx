import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import IAIPopover from 'common/components/IAIPopover';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { setBrushColor, setMaskColor } from 'features/canvas/store/canvasSlice';
import { clamp, isEqual } from 'lodash';

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
    if (layer === 'base')
      return `rgba(${brushColor.r},${brushColor.g},${brushColor.b},${brushColor.a})`;
    if (layer === 'mask')
      return `rgba(${maskColor.r},${maskColor.g},${maskColor.b},${maskColor.a})`;
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
      trigger="hover"
      triggerComponent={
        <Box
          style={{
            width: '30px',
            height: '30px',
            minWidth: '30px',
            minHeight: '30px',
            borderRadius: '99999999px',
            backgroundColor: currentColorDisplay(),
            cursor: 'pointer',
          }}
        />
      }
    >
      <Flex minWidth="15rem" direction="column" gap="1rem" width="100%">
        {layer === 'base' && (
          <IAIColorPicker
            style={{
              width: '100%',
              paddingTop: '0.5rem',
              paddingBottom: '0.5rem',
            }}
            color={brushColor}
            onChange={(newColor) => dispatch(setBrushColor(newColor))}
          />
        )}
        {layer === 'mask' && (
          <IAIColorPicker
            style={{
              width: '100%',
              paddingTop: '0.5rem',
              paddingBottom: '0.5rem',
            }}
            color={maskColor}
            onChange={(newColor) => dispatch(setMaskColor(newColor))}
          />
        )}
      </Flex>
    </IAIPopover>
  );
}

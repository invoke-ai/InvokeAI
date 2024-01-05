import { Box, Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import { InvButtonGroup } from 'common/components/InvButtonGroup/InvButtonGroup';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import {
  InvPopoverBody,
  InvPopoverContent,
  InvPopoverTrigger,
} from 'common/components/InvPopover/wrapper';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { resetToolInteractionState } from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  addEraseRect,
  addFillRect,
  selectCanvasSlice,
  setBrushColor,
  setBrushSize,
  setTool,
} from 'features/canvas/store/canvasSlice';
import { InvIconButton, InvPopover } from 'index';
import { clamp } from 'lodash-es';
import { memo, useCallback } from 'react';
import type { RgbaColor } from 'react-colorful';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  FaEraser,
  FaEyeDropper,
  FaFillDrip,
  FaPaintBrush,
  FaSlidersH,
  FaTimes,
} from 'react-icons/fa';

export const selector = createMemoizedSelector(
  [selectCanvasSlice, isStagingSelector],
  (canvas, isStaging) => {
    const { tool, brushColor, brushSize } = canvas;

    return {
      tool,
      isStaging,
      brushColor,
      brushSize,
    };
  }
);

const IAICanvasToolChooserOptions = () => {
  const dispatch = useAppDispatch();
  const { tool, brushColor, brushSize, isStaging } = useAppSelector(selector);
  const { t } = useTranslation();

  useHotkeys(
    ['b'],
    () => {
      handleSelectBrushTool();
    },
    {
      enabled: () => !isStaging,
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
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [tool]
  );

  useHotkeys(
    ['c'],
    () => {
      handleSelectColorPickerTool();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [tool]
  );

  useHotkeys(
    ['shift+f'],
    () => {
      handleFillRect();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    }
  );

  useHotkeys(
    ['delete', 'backspace'],
    () => {
      handleEraseBoundingBox();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    }
  );

  useHotkeys(
    ['BracketLeft'],
    () => {
      if (brushSize - 5 <= 5) {
        dispatch(setBrushSize(Math.max(brushSize - 1, 1)));
      } else {
        dispatch(setBrushSize(Math.max(brushSize - 5, 1)));
      }
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushSize]
  );

  useHotkeys(
    ['BracketRight'],
    () => {
      dispatch(setBrushSize(Math.min(brushSize + 5, 500)));
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [brushSize]
  );

  useHotkeys(
    ['Shift+BracketLeft'],
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
    ['Shift+BracketRight'],
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

  const handleSelectBrushTool = useCallback(() => {
    dispatch(setTool('brush'));
    resetToolInteractionState();
  }, [dispatch]);
  const handleSelectEraserTool = useCallback(() => {
    dispatch(setTool('eraser'));
    resetToolInteractionState();
  }, [dispatch]);
  const handleSelectColorPickerTool = useCallback(() => {
    dispatch(setTool('colorPicker'));
    resetToolInteractionState();
  }, [dispatch]);
  const handleFillRect = useCallback(() => {
    dispatch(addFillRect());
  }, [dispatch]);
  const handleEraseBoundingBox = useCallback(() => {
    dispatch(addEraseRect());
  }, [dispatch]);
  const handleChangeBrushSize = useCallback(
    (newSize: number) => {
      dispatch(setBrushSize(newSize));
    },
    [dispatch]
  );
  const handleChangeBrushColor = useCallback(
    (newColor: RgbaColor) => {
      dispatch(setBrushColor(newColor));
    },
    [dispatch]
  );

  return (
    <InvButtonGroup>
      <InvIconButton
        aria-label={`${t('unifiedCanvas.brush')} (B)`}
        tooltip={`${t('unifiedCanvas.brush')} (B)`}
        icon={<FaPaintBrush />}
        isChecked={tool === 'brush' && !isStaging}
        onClick={handleSelectBrushTool}
        isDisabled={isStaging}
      />
      <InvIconButton
        aria-label={`${t('unifiedCanvas.eraser')} (E)`}
        tooltip={`${t('unifiedCanvas.eraser')} (E)`}
        icon={<FaEraser />}
        isChecked={tool === 'eraser' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectEraserTool}
      />
      <InvIconButton
        aria-label={`${t('unifiedCanvas.fillBoundingBox')} (Shift+F)`}
        tooltip={`${t('unifiedCanvas.fillBoundingBox')} (Shift+F)`}
        icon={<FaFillDrip />}
        isDisabled={isStaging}
        onClick={handleFillRect}
      />
      <InvIconButton
        aria-label={`${t('unifiedCanvas.eraseBoundingBox')} (Del/Backspace)`}
        tooltip={`${t('unifiedCanvas.eraseBoundingBox')} (Del/Backspace)`}
        icon={<FaTimes />}
        isDisabled={isStaging}
        onClick={handleEraseBoundingBox}
      />
      <InvIconButton
        aria-label={`${t('unifiedCanvas.colorPicker')} (C)`}
        tooltip={`${t('unifiedCanvas.colorPicker')} (C)`}
        icon={<FaEyeDropper />}
        isChecked={tool === 'colorPicker' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectColorPickerTool}
      />
      <InvPopover>
        <InvPopoverTrigger>
          <InvIconButton
            aria-label={t('unifiedCanvas.brushOptions')}
            tooltip={t('unifiedCanvas.brushOptions')}
            icon={<FaSlidersH />}
          />
        </InvPopoverTrigger>
        <InvPopoverContent>
          <InvPopoverBody>
            <Flex minWidth={60} direction="column" gap={4} width="100%">
              <Flex gap={4} justifyContent="space-between">
                <InvControl label={t('unifiedCanvas.brushSize')}>
                  <InvSlider
                    value={brushSize}
                    min={1}
                    max={100}
                    step={1}
                    onChange={handleChangeBrushSize}
                    marks={marks}
                    defaultValue={50}
                  />
                  <InvNumberInput
                    value={brushSize}
                    min={1}
                    max={500}
                    step={1}
                    onChange={handleChangeBrushSize}
                    defaultValue={50}
                  />
                </InvControl>
              </Flex>
              <Box w="full" pt={2} pb={2}>
                <IAIColorPicker
                  withNumberInput={true}
                  color={brushColor}
                  onChange={handleChangeBrushColor}
                />
              </Box>
            </Flex>
          </InvPopoverBody>
        </InvPopoverContent>
      </InvPopover>
    </InvButtonGroup>
  );
};

export default memo(IAICanvasToolChooserOptions);

const marks = [1, 25, 50, 75, 100];

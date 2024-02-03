import {
  Box,
  ButtonGroup,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import { $tool, resetToolInteractionState } from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { addEraseRect, addFillRect, setBrushColor, setBrushSize } from 'features/canvas/store/canvasSlice';
import { clamp } from 'lodash-es';
import { memo, useCallback } from 'react';
import type { RgbaColor } from 'react-colorful';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  PiEraserBold,
  PiEyedropperBold,
  PiPaintBrushBold,
  PiPaintBucketBold,
  PiSlidersHorizontalBold,
  PiXBold,
} from 'react-icons/pi';

const marks = [1, 25, 50, 75, 100];

const IAICanvasToolChooserOptions = () => {
  const dispatch = useAppDispatch();
  const tool = useStore($tool);
  const brushColor = useAppSelector((s) => s.canvas.brushColor);
  const brushSize = useAppSelector((s) => s.canvas.brushSize);
  const isStaging = useAppSelector(isStagingSelector);
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
    $tool.set('brush');
    resetToolInteractionState();
  }, []);
  const handleSelectEraserTool = useCallback(() => {
    $tool.set('eraser');
    resetToolInteractionState();
  }, []);
  const handleSelectColorPickerTool = useCallback(() => {
    $tool.set('colorPicker');
    resetToolInteractionState();
  }, []);
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
    <ButtonGroup>
      <IconButton
        aria-label={`${t('unifiedCanvas.brush')} (B)`}
        tooltip={`${t('unifiedCanvas.brush')} (B)`}
        icon={<PiPaintBrushBold />}
        isChecked={tool === 'brush' && !isStaging}
        onClick={handleSelectBrushTool}
        isDisabled={isStaging}
      />
      <IconButton
        aria-label={`${t('unifiedCanvas.eraser')} (E)`}
        tooltip={`${t('unifiedCanvas.eraser')} (E)`}
        icon={<PiEraserBold />}
        isChecked={tool === 'eraser' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectEraserTool}
      />
      <IconButton
        aria-label={`${t('unifiedCanvas.fillBoundingBox')} (Shift+F)`}
        tooltip={`${t('unifiedCanvas.fillBoundingBox')} (Shift+F)`}
        icon={<PiPaintBucketBold />}
        isDisabled={isStaging}
        onClick={handleFillRect}
      />
      <IconButton
        aria-label={`${t('unifiedCanvas.eraseBoundingBox')} (Del/Backspace)`}
        tooltip={`${t('unifiedCanvas.eraseBoundingBox')} (Del/Backspace)`}
        icon={<PiXBold />}
        isDisabled={isStaging}
        onClick={handleEraseBoundingBox}
      />
      <IconButton
        aria-label={`${t('unifiedCanvas.colorPicker')} (C)`}
        tooltip={`${t('unifiedCanvas.colorPicker')} (C)`}
        icon={<PiEyedropperBold />}
        isChecked={tool === 'colorPicker' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectColorPickerTool}
      />
      <Popover>
        <PopoverTrigger>
          <IconButton
            aria-label={t('unifiedCanvas.brushOptions')}
            tooltip={t('unifiedCanvas.brushOptions')}
            icon={<PiSlidersHorizontalBold />}
          />
        </PopoverTrigger>
        <PopoverContent>
          <PopoverBody>
            <Flex minWidth={60} direction="column" gap={4} width="100%">
              <Flex gap={4} justifyContent="space-between">
                <FormControl>
                  <FormLabel>{t('unifiedCanvas.brushSize')}</FormLabel>
                  <CompositeSlider
                    value={brushSize}
                    min={1}
                    max={100}
                    step={1}
                    onChange={handleChangeBrushSize}
                    marks={marks}
                    defaultValue={50}
                  />
                  <CompositeNumberInput
                    value={brushSize}
                    min={1}
                    max={500}
                    step={1}
                    onChange={handleChangeBrushSize}
                    defaultValue={50}
                  />
                </FormControl>
              </Flex>
              <Box w="full" pt={2} pb={2}>
                <IAIColorPicker color={brushColor} onChange={handleChangeBrushColor} withNumberInput />
              </Box>
            </Flex>
          </PopoverBody>
        </PopoverContent>
      </Popover>
    </ButtonGroup>
  );
};

export default memo(IAICanvasToolChooserOptions);

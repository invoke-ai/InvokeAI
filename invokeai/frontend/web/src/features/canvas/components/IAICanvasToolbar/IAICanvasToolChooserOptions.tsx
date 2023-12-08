import { Box, ButtonGroup, Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import IAISlider from 'common/components/IAISlider';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  addEraseRect,
  addFillRect,
  setBrushColor,
  setBrushSize,
  setTool,
} from 'features/canvas/store/canvasSlice';
import { clamp } from 'lodash-es';
import { memo, useCallback } from 'react';
import { RgbaColor } from 'react-colorful';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  FaEraser,
  FaEyeDropper,
  FaFillDrip,
  FaPaintBrush,
  FaPlus,
  FaSlidersH,
} from 'react-icons/fa';

export const selector = createMemoizedSelector(
  [stateSelector, isStagingSelector],
  ({ canvas }, isStaging) => {
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
  }, [dispatch]);
  const handleSelectEraserTool = useCallback(() => {
    dispatch(setTool('eraser'));
  }, [dispatch]);
  const handleSelectColorPickerTool = useCallback(() => {
    dispatch(setTool('colorPicker'));
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
    <ButtonGroup isAttached>
      <IAIIconButton
        aria-label={`${t('unifiedCanvas.brush')} (B)`}
        tooltip={`${t('unifiedCanvas.brush')} (B)`}
        icon={<FaPaintBrush />}
        isChecked={tool === 'brush' && !isStaging}
        onClick={handleSelectBrushTool}
        isDisabled={isStaging}
      />
      <IAIIconButton
        aria-label={`${t('unifiedCanvas.eraser')} (E)`}
        tooltip={`${t('unifiedCanvas.eraser')} (E)`}
        icon={<FaEraser />}
        isChecked={tool === 'eraser' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectEraserTool}
      />
      <IAIIconButton
        aria-label={`${t('unifiedCanvas.fillBoundingBox')} (Shift+F)`}
        tooltip={`${t('unifiedCanvas.fillBoundingBox')} (Shift+F)`}
        icon={<FaFillDrip />}
        isDisabled={isStaging}
        onClick={handleFillRect}
      />
      <IAIIconButton
        aria-label={`${t('unifiedCanvas.eraseBoundingBox')} (Del/Backspace)`}
        tooltip={`${t('unifiedCanvas.eraseBoundingBox')} (Del/Backspace)`}
        icon={<FaPlus style={{ transform: 'rotate(45deg)' }} />}
        isDisabled={isStaging}
        onClick={handleEraseBoundingBox}
      />
      <IAIIconButton
        aria-label={`${t('unifiedCanvas.colorPicker')} (C)`}
        tooltip={`${t('unifiedCanvas.colorPicker')} (C)`}
        icon={<FaEyeDropper />}
        isChecked={tool === 'colorPicker' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectColorPickerTool}
      />
      <IAIPopover
        triggerComponent={
          <IAIIconButton
            aria-label={t('unifiedCanvas.brushOptions')}
            tooltip={t('unifiedCanvas.brushOptions')}
            icon={<FaSlidersH />}
          />
        }
      >
        <Flex minWidth={60} direction="column" gap={4} width="100%">
          <Flex gap={4} justifyContent="space-between">
            <IAISlider
              label={t('unifiedCanvas.brushSize')}
              value={brushSize}
              withInput
              onChange={handleChangeBrushSize}
              sliderNumberInputProps={{ max: 500 }}
            />
          </Flex>
          <Box
            sx={{
              width: '100%',
              paddingTop: 2,
              paddingBottom: 2,
            }}
          >
            <IAIColorPicker
              withNumberInput={true}
              color={brushColor}
              onChange={handleChangeBrushColor}
            />
          </Box>
        </Flex>
      </IAIPopover>
    </ButtonGroup>
  );
};

export default memo(IAICanvasToolChooserOptions);

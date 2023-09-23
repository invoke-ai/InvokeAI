import { ButtonGroup, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  addEraseRect,
  addFillRect,
  setTool,
} from 'features/canvas/store/canvasSlice';
import { isEqual } from 'lodash-es';
import { memo, useCallback } from 'react';

import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  FaEraser,
  FaEyeDropper,
  FaFillDrip,
  FaPaintBrush,
  FaPlus,
} from 'react-icons/fa';

export const selector = createSelector(
  [stateSelector, isStagingSelector],
  ({ canvas }, isStaging) => {
    const { tool } = canvas;

    return {
      tool,
      isStaging,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const UnifiedCanvasToolSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { tool, isStaging } = useAppSelector(selector);

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

  const handleSelectBrushTool = useCallback(
    () => dispatch(setTool('brush')),
    [dispatch]
  );
  const handleSelectEraserTool = useCallback(
    () => dispatch(setTool('eraser')),
    [dispatch]
  );
  const handleSelectColorPickerTool = useCallback(
    () => dispatch(setTool('colorPicker')),
    [dispatch]
  );
  const handleFillRect = useCallback(() => dispatch(addFillRect()), [dispatch]);
  const handleEraseBoundingBox = useCallback(
    () => dispatch(addEraseRect()),
    [dispatch]
  );

  return (
    <Flex flexDirection="column" gap={2}>
      <ButtonGroup>
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
          tooltip={`${t('unifiedCanvas.eraser')} (B)`}
          icon={<FaEraser />}
          isChecked={tool === 'eraser' && !isStaging}
          isDisabled={isStaging}
          onClick={handleSelectEraserTool}
        />
      </ButtonGroup>
      <ButtonGroup>
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
      </ButtonGroup>
      <IAIIconButton
        aria-label={`${t('unifiedCanvas.colorPicker')} (C)`}
        tooltip={`${t('unifiedCanvas.colorPicker')} (C)`}
        icon={<FaEyeDropper />}
        isChecked={tool === 'colorPicker' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectColorPickerTool}
        width="max-content"
      />
    </Flex>
  );
};

export default memo(UnifiedCanvasToolSelect);

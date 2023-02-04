import { ButtonGroup, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import {
  addEraseRect,
  addFillRect,
  setTool,
} from 'features/canvas/store/canvasSlice';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import _ from 'lodash';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  FaEraser,
  FaEyeDropper,
  FaFillDrip,
  FaPaintBrush,
  FaPlus,
} from 'react-icons/fa';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export const selector = createSelector(
  [canvasSelector, isStagingSelector, systemSelector],
  (canvas, isStaging, system) => {
    const { isProcessing } = system;
    const { tool } = canvas;

    return {
      tool,
      isStaging,
      isProcessing,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
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

  const handleSelectBrushTool = () => dispatch(setTool('brush'));
  const handleSelectEraserTool = () => dispatch(setTool('eraser'));
  const handleSelectColorPickerTool = () => dispatch(setTool('colorPicker'));
  const handleFillRect = () => dispatch(addFillRect());
  const handleEraseBoundingBox = () => dispatch(addEraseRect());

  return (
    <Flex flexDirection={'column'} gap={'0.5rem'}>
      <ButtonGroup>
        <IAIIconButton
          aria-label={`${t('unifiedcanvas:brush')} (B)`}
          tooltip={`${t('unifiedcanvas:brush')} (B)`}
          icon={<FaPaintBrush />}
          data-selected={tool === 'brush' && !isStaging}
          onClick={handleSelectBrushTool}
          isDisabled={isStaging}
        />
        <IAIIconButton
          aria-label={`${t('unifiedcanvas:eraser')} (E)`}
          tooltip={`${t('unifiedcanvas:eraser')} (B)`}
          icon={<FaEraser />}
          data-selected={tool === 'eraser' && !isStaging}
          isDisabled={isStaging}
          onClick={handleSelectEraserTool}
        />
      </ButtonGroup>
      <ButtonGroup>
        <IAIIconButton
          aria-label={`${t('unifiedcanvas:fillBoundingBox')} (Shift+F)`}
          tooltip={`${t('unifiedcanvas:fillBoundingBox')} (Shift+F)`}
          icon={<FaFillDrip />}
          isDisabled={isStaging}
          onClick={handleFillRect}
        />
        <IAIIconButton
          aria-label={`${t('unifiedcanvas:eraseBoundingBox')} (Del/Backspace)`}
          tooltip={`${t('unifiedcanvas:eraseBoundingBox')} (Del/Backspace)`}
          icon={<FaPlus style={{ transform: 'rotate(45deg)' }} />}
          isDisabled={isStaging}
          onClick={handleEraseBoundingBox}
        />
      </ButtonGroup>
      <IAIIconButton
        aria-label={`${t('unifiedcanvas:colorPicker')} (C)`}
        tooltip={`${t('unifiedcanvas:colorPicker')} (C)`}
        icon={<FaEyeDropper />}
        data-selected={tool === 'colorPicker' && !isStaging}
        isDisabled={isStaging}
        onClick={handleSelectColorPickerTool}
        width={'max-content'}
      />
    </Flex>
  );
};

export default UnifiedCanvasToolSelect;

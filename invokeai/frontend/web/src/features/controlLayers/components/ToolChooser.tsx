import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  $tool,
  selectedLayerDeleted,
  selectedLayerReset,
  selectRegionalPromptsSlice,
} from 'features/controlLayers/store/regionalPromptsSlice';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutCardinalBold, PiEraserBold, PiPaintBrushBold, PiRectangleBold } from 'react-icons/pi';

const selectIsDisabled = createSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  const selectedLayer = regionalPrompts.present.layers.find((l) => l.id === regionalPrompts.present.selectedLayerId);
  return selectedLayer?.type !== 'masked_guidance_layer';
});

export const ToolChooser: React.FC = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isDisabled = useAppSelector(selectIsDisabled);
  const tool = useStore($tool);

  const setToolToBrush = useCallback(() => {
    $tool.set('brush');
  }, []);
  useHotkeys('b', setToolToBrush, { enabled: !isDisabled }, [isDisabled]);
  const setToolToEraser = useCallback(() => {
    $tool.set('eraser');
  }, []);
  useHotkeys('e', setToolToEraser, { enabled: !isDisabled }, [isDisabled]);
  const setToolToRect = useCallback(() => {
    $tool.set('rect');
  }, []);
  useHotkeys('u', setToolToRect, { enabled: !isDisabled }, [isDisabled]);
  const setToolToMove = useCallback(() => {
    $tool.set('move');
  }, []);
  useHotkeys('v', setToolToMove, { enabled: !isDisabled }, [isDisabled]);

  const resetSelectedLayer = useCallback(() => {
    dispatch(selectedLayerReset());
  }, [dispatch]);
  useHotkeys('shift+c', resetSelectedLayer);

  const deleteSelectedLayer = useCallback(() => {
    dispatch(selectedLayerDeleted());
  }, [dispatch]);
  useHotkeys('shift+d', deleteSelectedLayer);

  return (
    <ButtonGroup isAttached>
      <IconButton
        aria-label={`${t('unifiedCanvas.brush')} (B)`}
        tooltip={`${t('unifiedCanvas.brush')} (B)`}
        icon={<PiPaintBrushBold />}
        variant={tool === 'brush' ? 'solid' : 'outline'}
        onClick={setToolToBrush}
        isDisabled={isDisabled}
      />
      <IconButton
        aria-label={`${t('unifiedCanvas.eraser')} (E)`}
        tooltip={`${t('unifiedCanvas.eraser')} (E)`}
        icon={<PiEraserBold />}
        variant={tool === 'eraser' ? 'solid' : 'outline'}
        onClick={setToolToEraser}
        isDisabled={isDisabled}
      />
      <IconButton
        aria-label={`${t('regionalPrompts.rectangle')} (U)`}
        tooltip={`${t('regionalPrompts.rectangle')} (U)`}
        icon={<PiRectangleBold />}
        variant={tool === 'rect' ? 'solid' : 'outline'}
        onClick={setToolToRect}
        isDisabled={isDisabled}
      />
      <IconButton
        aria-label={`${t('unifiedCanvas.move')} (V)`}
        tooltip={`${t('unifiedCanvas.move')} (V)`}
        icon={<PiArrowsOutCardinalBold />}
        variant={tool === 'move' ? 'solid' : 'outline'}
        onClick={setToolToMove}
        isDisabled={isDisabled}
      />
    </ButtonGroup>
  );
};

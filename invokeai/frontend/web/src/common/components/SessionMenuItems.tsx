import { MenuItem } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  allEntitiesDeleted,
  bboxAspectRatioLockToggled,
  bboxSizeOptimized,
} from 'features/controlLayers/store/canvasSlice';
import { paramsReset } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold } from 'react-icons/pi';

const selectAspectRatioIsLocked = createSelector(selectCanvasSlice, (canvas) => canvas.bbox.aspectRatio.isLocked);

export const SessionMenuItems = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManagerSafe();
  const isAspectRatioLocked = useAppSelector(selectAspectRatioIsLocked);

  const resetCanvasLayers = useCallback(() => {
    dispatch(allEntitiesDeleted());
  }, [dispatch]);

  const resetGenerationSettings = useCallback(() => {
    // Reset the generation parameters
    dispatch(paramsReset());

    // Unlock aspect ratio if it's currently locked
    if (isAspectRatioLocked) {
      dispatch(bboxAspectRatioLockToggled());
    }

    // Optimize size for model
    dispatch(bboxSizeOptimized());

    // Fit layers to canvas if canvas manager is available
    if (canvasManager) {
      canvasManager.stage.fitLayersToStage();
    }
  }, [dispatch, isAspectRatioLocked, canvasManager]);

  return (
    <>
      <MenuItem icon={<PiArrowsCounterClockwiseBold />} onClick={resetCanvasLayers}>
        {t('controlLayers.resetCanvasLayers')}
      </MenuItem>
      <MenuItem icon={<PiArrowsCounterClockwiseBold />} onClick={resetGenerationSettings}>
        {t('controlLayers.resetGenerationSettings')}
      </MenuItem>
    </>
  );
});

SessionMenuItems.displayName = 'SessionMenuItems';

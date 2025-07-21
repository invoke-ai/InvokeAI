import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { canvasReset } from 'features/controlLayers/store/actions';
import { inpaintMaskAdded } from 'features/controlLayers/store/canvasSlice';
import { $canvasManager } from 'features/controlLayers/store/ephemeral';
import { paramsReset } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold } from 'react-icons/pi';

export const SessionMenuItems = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const resetCanvasLayers = useCallback(() => {
    dispatch(canvasReset());
    dispatch(inpaintMaskAdded({ isSelected: true, isBookmarked: true }));
    $canvasManager.get()?.stage.fitBboxToStage();
  }, [dispatch]);
  const resetGenerationSettings = useCallback(() => {
    dispatch(paramsReset());
  }, [dispatch]);
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

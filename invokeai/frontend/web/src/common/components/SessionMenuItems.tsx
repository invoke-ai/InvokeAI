import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { allEntitiesDeleted, inpaintMaskAdded } from 'features/controlLayers/store/canvasInstanceSlice';
import { paramsReset } from 'features/controlLayers/store/paramsSlice';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold } from 'react-icons/pi';

export const SessionMenuItems = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const tab = useAppSelector(selectActiveTab);

  const canvasManager = useCanvasManagerSafe();
  const resetCanvasLayers = useCallback(() => {
    dispatch(allEntitiesDeleted());
    dispatch(inpaintMaskAdded({ isSelected: true, isBookmarked: true }));
    canvasManager?.stage.fitBboxToStage();
  }, [dispatch, canvasManager]);
  const resetGenerationSettings = useCallback(() => {
    dispatch(paramsReset());
  }, [dispatch]);
  return (
    <>
      {tab === 'canvas' && (
        <MenuItem icon={<PiArrowsCounterClockwiseBold />} onClick={resetCanvasLayers}>
          {t('controlLayers.resetCanvasLayers')}
        </MenuItem>
      )}
      {(tab === 'canvas' || tab === 'generate') && (
        <MenuItem icon={<PiArrowsCounterClockwiseBold />} onClick={resetGenerationSettings}>
          {t('controlLayers.resetGenerationSettings')}
        </MenuItem>
      )}
    </>
  );
});

SessionMenuItems.displayName = 'SessionMenuItems';

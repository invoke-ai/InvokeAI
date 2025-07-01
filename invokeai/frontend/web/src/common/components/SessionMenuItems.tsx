import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { allEntitiesDeleted } from 'features/controlLayers/store/canvasSlice';
import { paramsReset } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold, PiFilePlusBold } from 'react-icons/pi';

export const SessionMenuItems = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const resetCanvasLayers = useCallback(() => {
    dispatch(allEntitiesDeleted());
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

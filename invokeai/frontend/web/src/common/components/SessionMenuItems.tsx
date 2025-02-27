import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import {
  useNewCanvasSession,
  useNewGallerySession,
} from 'features/controlLayers/components/NewSessionConfirmationAlertDialog';
import { allEntitiesDeleted } from 'features/controlLayers/store/canvasSlice';
import { paramsReset } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold, PiFilePlusBold } from 'react-icons/pi';

export const SessionMenuItems = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { newGallerySessionWithDialog } = useNewGallerySession();
  const { newCanvasSessionWithDialog } = useNewCanvasSession();
  const resetCanvasLayers = useCallback(() => {
    dispatch(allEntitiesDeleted());
  }, [dispatch]);
  const resetGenerationSettings = useCallback(() => {
    dispatch(paramsReset());
  }, [dispatch]);
  return (
    <>
      <MenuItem icon={<PiFilePlusBold />} onClick={newGallerySessionWithDialog}>
        {t('controlLayers.newGallerySession')}
      </MenuItem>
      <MenuItem icon={<PiFilePlusBold />} onClick={newCanvasSessionWithDialog}>
        {t('controlLayers.newCanvasSession')}
      </MenuItem>
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

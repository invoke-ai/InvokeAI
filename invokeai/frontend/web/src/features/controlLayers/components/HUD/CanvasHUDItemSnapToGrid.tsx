import { useAppSelector } from 'app/store/storeHooks';
import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { selectSnapToGrid } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasHUDItemSnapToGrid = memo(() => {
  const { t } = useTranslation();
  const snapToGrid = useAppSelector(selectSnapToGrid);

  return (
    <CanvasHUDItem
      label={t('controlLayers.settings.snapToGrid.label')}
      value={snapToGrid ? t('common.on') : t('common.off')}
    />
  );
});

CanvasHUDItemSnapToGrid.displayName = 'CanvasHUDItemSnapToGrid';

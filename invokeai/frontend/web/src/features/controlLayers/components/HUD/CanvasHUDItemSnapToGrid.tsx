import { useAppSelector } from 'app/store/storeHooks';
import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { selectSnapToGrid } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasHUDItemSnapToGrid = memo(() => {
  const { t } = useTranslation();
  const snap = useAppSelector(selectSnapToGrid);
  const snapString = useMemo(() => {
    switch (snap) {
      case 'off':
        return t('controlLayers.settings.snapToGrid.off');
      case '8':
        return t('controlLayers.settings.snapToGrid.8');
      case '64':
        return t('controlLayers.settings.snapToGrid.64');
    }
  }, [snap, t]);

  return <CanvasHUDItem label={t('controlLayers.HUD.snapToGrid')} value={snapString} />;
});

CanvasHUDItemSnapToGrid.displayName = 'CanvasHUDItemSnapToGrid';

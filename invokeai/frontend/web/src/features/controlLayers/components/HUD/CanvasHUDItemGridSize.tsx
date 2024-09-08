import { useAppSelector } from 'app/store/storeHooks';
import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { selectGridSize } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasHUDItemGridSize = memo(() => {
  const { t } = useTranslation();
  const snap = useAppSelector(selectGridSize);
  const snapString = useMemo(() => {
    switch (snap) {
      case 1:
        return t('controlLayers.settings.snapToGrid.off');
      case 8:
        return t('controlLayers.settings.snapToGrid.8');
      case 64:
        return t('controlLayers.settings.snapToGrid.64');
    }
  }, [snap, t]);

  return <CanvasHUDItem label={t('controlLayers.settings.snapToGrid.label')} value={snapString} />;
});

CanvasHUDItemGridSize.displayName = 'CanvasHUDItemGridSize';

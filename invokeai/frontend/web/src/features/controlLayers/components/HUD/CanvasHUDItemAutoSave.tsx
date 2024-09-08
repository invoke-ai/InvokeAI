import { useAppSelector } from 'app/store/storeHooks';
import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { selectAutoSave } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasHUDItemAutoSave = memo(() => {
  const { t } = useTranslation();
  const autoSave = useAppSelector(selectAutoSave);

  return <CanvasHUDItem label={t('controlLayers.HUD.autoSave')} value={autoSave ? t('common.on') : t('common.off')} />;
});

CanvasHUDItemAutoSave.displayName = 'CanvasHUDItemAutoSave';

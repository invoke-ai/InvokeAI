import { useAppSelector } from 'app/store/storeHooks';
import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { selectScaledSize, selectScaleMethod } from 'features/controlLayers/store/selectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasHUDItemScaledBbox = memo(() => {
  const { t } = useTranslation();
  const scaleMethod = useAppSelector(selectScaleMethod);
  const scaledSize = useAppSelector(selectScaledSize);

  if (scaleMethod === 'none') {
    return null;
  }

  return (
    <CanvasHUDItem label={t('controlLayers.HUD.scaledBbox')} value={`${scaledSize.width}×${scaledSize.height} px`} />
  );
});

CanvasHUDItemScaledBbox.displayName = 'CanvasHUDItemScaledBbox';

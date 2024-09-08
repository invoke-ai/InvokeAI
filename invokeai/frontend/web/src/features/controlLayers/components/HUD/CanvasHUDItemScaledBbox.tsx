import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { selectBbox } from 'features/controlLayers/store/selectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectScaledSize = createSelector(selectBbox, (bbox) => bbox.scaledSize);

export const CanvasHUDItemScaledBbox = memo(() => {
  const { t } = useTranslation();
  const scaledSize = useAppSelector(selectScaledSize);

  return (
    <CanvasHUDItem label={t('controlLayers.HUD.scaledBbox')} value={`${scaledSize.width}×${scaledSize.height} px`} />
  );
});

CanvasHUDItemScaledBbox.displayName = 'CanvasHUDItemScaledBbox';

import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasHUDItem } from 'features/controlLayers/components/HUD/CanvasHUDItem';
import { selectBbox } from 'features/controlLayers/store/selectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selectBboxRect = createSelector(selectBbox, (bbox) => bbox.rect);

export const CanvasHUDItemBbox = memo(() => {
  const { t } = useTranslation();
  const rect = useAppSelector(selectBboxRect);

  return <CanvasHUDItem label={t('controlLayers.HUD.bbox')} value={`${rect.width}×${rect.height} px`} />;
});

CanvasHUDItemBbox.displayName = 'CanvasHUDItemBbox';

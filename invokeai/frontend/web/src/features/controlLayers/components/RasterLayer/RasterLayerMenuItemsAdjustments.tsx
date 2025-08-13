import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerAdjustmentsReset, rasterLayerAdjustmentsSet } from 'features/controlLayers/store/canvasSlice';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSlidersHorizontalBold } from 'react-icons/pi';

export const RasterLayerMenuItemsAdjustments = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const { t } = useTranslation();
  const layer = useAppSelector((s) =>
    s.canvas.present.rasterLayers.entities.find((e: CanvasRasterLayerState) => e.id === entityIdentifier.id)
  );
  const hasAdjustments = Boolean(layer?.adjustments);
  const onToggleAdjustmentsPresence = useCallback(() => {
    if (hasAdjustments) {
      dispatch(rasterLayerAdjustmentsReset({ entityIdentifier }));
    } else {
      dispatch(
        rasterLayerAdjustmentsSet({
          entityIdentifier,
          adjustments: { enabled: true, collapsed: false, mode: 'simple' },
        })
      );
    }
  }, [dispatch, entityIdentifier, hasAdjustments]);

  return (
    <MenuItem onClick={onToggleAdjustmentsPresence} icon={<PiSlidersHorizontalBold />}>
      {hasAdjustments ? t('controlLayers.removeAdjustments') : t('controlLayers.addAdjustments')}
    </MenuItem>
  );
});

RasterLayerMenuItemsAdjustments.displayName = 'RasterLayerMenuItemsAdjustments';

import { MenuItem } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerAdjustmentsCancel, rasterLayerAdjustmentsSet } from 'features/controlLayers/store/canvasSlice';
import { selectActiveCanvas } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { makeDefaultRasterLayerAdjustments } from 'features/controlLayers/store/util';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSlidersHorizontalBold } from 'react-icons/pi';

export const RasterLayerMenuItemsAdjustments = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const { t } = useTranslation();
  const selectRasterLayer = useMemo(() => {
    return createSelector(selectActiveCanvas, (canvas) =>
      canvas.rasterLayers.entities.find((e: CanvasRasterLayerState) => e.id === entityIdentifier.id)
    );
  }, [entityIdentifier]);
  const layer = useAppSelector(selectRasterLayer);
  const hasAdjustments = Boolean(layer?.adjustments);
  const onToggleAdjustmentsPresence = useCallback(() => {
    if (hasAdjustments) {
      dispatch(rasterLayerAdjustmentsCancel({ entityIdentifier }));
    } else {
      dispatch(
        rasterLayerAdjustmentsSet({
          entityIdentifier,
          adjustments: makeDefaultRasterLayerAdjustments('simple'),
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

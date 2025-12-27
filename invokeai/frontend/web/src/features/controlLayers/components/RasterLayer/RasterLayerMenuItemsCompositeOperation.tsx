import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerGlobalCompositeOperationChanged } from 'features/controlLayers/store/canvasSlice';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDropHalfBold } from 'react-icons/pi';

export const RasterLayerMenuItemsCompositeOperation = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const { t } = useTranslation();
  const layer = useAppSelector((s) =>
    s.canvas.present.rasterLayers.entities.find((e: CanvasRasterLayerState) => e.id === entityIdentifier.id)
  );
  const hasCompositeOperation = layer?.globalCompositeOperation !== undefined;

  const onToggleCompositeOperationPresence = useCallback(() => {
    if (hasCompositeOperation) {
      dispatch(rasterLayerGlobalCompositeOperationChanged({ entityIdentifier, globalCompositeOperation: undefined }));
    } else {
      // default to color when enabling blend modes
      dispatch(rasterLayerGlobalCompositeOperationChanged({ entityIdentifier, globalCompositeOperation: 'color' }));
    }
  }, [dispatch, entityIdentifier, hasCompositeOperation]);

  return (
    <MenuItem onClick={onToggleCompositeOperationPresence} icon={<PiDropHalfBold />}>
      {hasCompositeOperation ? t('controlLayers.compositeOperation.remove') : t('controlLayers.compositeOperation.add')}
    </MenuItem>
  );
});

RasterLayerMenuItemsCompositeOperation.displayName = 'RasterLayerMenuItemsCompositeOperation';

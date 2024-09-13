import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectDefaultControlAdapter } from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { rasterLayerConvertedToControlLayer } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightningBold } from 'react-icons/pi';

export const RasterLayerMenuItemsRasterToControl = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('raster_layer');
  const defaultControlAdapter = useAppSelector(selectDefaultControlAdapter);
  const isBusy = useCanvasIsBusy();

  const convertRasterLayerToControlLayer = useCallback(() => {
    dispatch(
      rasterLayerConvertedToControlLayer({
        entityIdentifier,
        overrides: {
          controlAdapter: defaultControlAdapter,
        },
      })
    );
  }, [defaultControlAdapter, dispatch, entityIdentifier]);

  return (
    <MenuItem onClick={convertRasterLayerToControlLayer} icon={<PiLightningBold />} isDisabled={isBusy}>
      {t('controlLayers.convertToControlLayer')}
    </MenuItem>
  );
});

RasterLayerMenuItemsRasterToControl.displayName = 'RasterLayerMenuItemsRasterToControl';

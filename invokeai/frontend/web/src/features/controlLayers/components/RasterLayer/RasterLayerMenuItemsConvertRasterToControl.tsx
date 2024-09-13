import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectDefaultControlAdapter } from 'features/controlLayers/hooks/addLayerHooks';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import { rasterLayerConvertedToControlLayer } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightningBold } from 'react-icons/pi';

export const RasterLayerMenuItemsConvertRasterToControl = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('raster_layer');
  const defaultControlAdapter = useAppSelector(selectDefaultControlAdapter);
  const isInteractable = useIsEntityInteractable(entityIdentifier);

  const onClick = useCallback(() => {
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
    <MenuItem onClick={onClick} icon={<PiLightningBold />} isDisabled={!isInteractable}>
      {t('controlLayers.convertToControlLayer')}
    </MenuItem>
  );
});

RasterLayerMenuItemsConvertRasterToControl.displayName = 'RasterLayerMenuItemsConvertRasterToControl';

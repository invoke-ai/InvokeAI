import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useDefaultControlAdapter } from 'features/controlLayers/hooks/useLayerControlAdapter';
import { rasterLayerConvertedToControlLayer } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightningBold } from 'react-icons/pi';

export const RasterLayerMenuItemsRasterToControl = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();

  const defaultControlAdapter = useDefaultControlAdapter();

  const convertRasterLayerToControlLayer = useCallback(() => {
    dispatch(rasterLayerConvertedToControlLayer({ id: entityIdentifier.id, controlAdapter: defaultControlAdapter }));
  }, [dispatch, defaultControlAdapter, entityIdentifier.id]);

  return (
    <MenuItem onClick={convertRasterLayerToControlLayer} icon={<PiLightningBold />}>
      {t('controlLayers.convertToControlLayer')}
    </MenuItem>
  );
});

RasterLayerMenuItemsRasterToControl.displayName = 'RasterLayerMenuItemsRasterToControl';

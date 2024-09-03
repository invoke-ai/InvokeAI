import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { controlLayerConvertedToRasterLayer } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightningBold } from 'react-icons/pi';

export const ControlLayerMenuItemsControlToRaster = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const entityIdentifier = useEntityIdentifierContext('control_layer');

  const convertControlLayerToRasterLayer = useCallback(() => {
    dispatch(controlLayerConvertedToRasterLayer({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem onClick={convertControlLayerToRasterLayer} icon={<PiLightningBold />} isDisabled={isBusy}>
      {t('controlLayers.convertToRasterLayer')}
    </MenuItem>
  );
});

ControlLayerMenuItemsControlToRaster.displayName = 'ControlLayerMenuItemsControlToRaster';

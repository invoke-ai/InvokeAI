import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useIsEntityInteractable } from 'features/controlLayers/hooks/useEntityIsInteractable';
import { controlLayerConvertedToRasterLayer } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightningBold } from 'react-icons/pi';

export const ControlLayerMenuItemsConvertControlToRaster = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const isInteractable = useIsEntityInteractable(entityIdentifier);

  const convertControlLayerToRasterLayer = useCallback(() => {
    dispatch(controlLayerConvertedToRasterLayer({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem onClick={convertControlLayerToRasterLayer} icon={<PiLightningBold />} isDisabled={!isInteractable}>
      {t('controlLayers.convertToRasterLayer')}
    </MenuItem>
  );
});

ControlLayerMenuItemsConvertControlToRaster.displayName = 'ControlLayerMenuItemsConvertControlToRaster';

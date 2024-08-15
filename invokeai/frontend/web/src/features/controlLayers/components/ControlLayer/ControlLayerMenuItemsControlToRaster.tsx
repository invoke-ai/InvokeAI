import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { controlLayerConvertedToRasterLayer } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightningBold } from 'react-icons/pi';

export const ControlLayerMenuItemsControlToRaster = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();

  const convertControlLayerToRasterLayer = useCallback(() => {
    dispatch(controlLayerConvertedToRasterLayer({ id: entityIdentifier.id }));
  }, [dispatch, entityIdentifier.id]);

  return (
    <MenuItem onClick={convertControlLayerToRasterLayer} icon={<PiLightningBold />}>
      {t('controlLayers.convertToRasterLayer')}
    </MenuItem>
  );
});

ControlLayerMenuItemsControlToRaster.displayName = 'ControlLayerMenuItemsControlToRaster';

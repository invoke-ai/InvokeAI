import { MenuGroup, MenuItem } from '@invoke-ai/ui-library';
import {
  useNewControlLayerFromBbox,
  useNewGlobalIPAdapterFromBbox,
  useNewRasterLayerFromBbox,
  useNewRegionalIPAdapterFromBbox,
  useSaveBboxToGallery,
  useSaveCanvasToGallery,
} from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold, PiStackPlusFill } from 'react-icons/pi';

export const CanvasContextMenuGlobalMenuItems = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const saveCanvasToGallery = useSaveCanvasToGallery();
  const saveBboxToGallery = useSaveBboxToGallery();
  const saveBboxAsRegionalGuidanceIPAdapter = useNewRegionalIPAdapterFromBbox();
  const saveBboxAsIPAdapter = useNewGlobalIPAdapterFromBbox();
  const saveBboxAsRasterLayer = useNewRasterLayerFromBbox();
  const saveBboxAsControlLayer = useNewControlLayerFromBbox();

  return (
    <MenuGroup title={t('controlLayers.canvas')}>
      <MenuItem icon={<PiFloppyDiskBold />} isDisabled={isBusy} onClick={saveCanvasToGallery}>
        {t('controlLayers.saveCanvasToGallery')}
      </MenuItem>
      <MenuItem icon={<PiFloppyDiskBold />} isDisabled={isBusy} onClick={saveBboxToGallery}>
        {t('controlLayers.saveBboxToGallery')}
      </MenuItem>
      <MenuItem icon={<PiStackPlusFill />} isDisabled={isBusy} onClick={saveBboxAsIPAdapter}>
        {t('controlLayers.newGlobalIPAdapterFromBbox')}
      </MenuItem>
      <MenuItem icon={<PiStackPlusFill />} isDisabled={isBusy} onClick={saveBboxAsRegionalGuidanceIPAdapter}>
        {t('controlLayers.newRegionalIPAdapterFromBbox')}
      </MenuItem>
      <MenuItem icon={<PiStackPlusFill />} isDisabled={isBusy} onClick={saveBboxAsControlLayer}>
        {t('controlLayers.newControlLayerFromBbox')}
      </MenuItem>
      <MenuItem icon={<PiStackPlusFill />} isDisabled={isBusy} onClick={saveBboxAsRasterLayer}>
        {t('controlLayers.newRasterLayerFromBbox')}
      </MenuItem>
    </MenuGroup>
  );
});

CanvasContextMenuGlobalMenuItems.displayName = 'CanvasContextMenuGlobalMenuItems';

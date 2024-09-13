import { MenuItem } from '@invoke-ai/ui-library';
import {
  useSaveBboxAsControlLayer,
  useSaveBboxAsGlobalIPAdapter,
  useSaveBboxAsRasterLayer,
  useSaveBboxAsRegionalGuidanceIPAdapter,
  useSaveBboxToGallery,
  useSaveCanvasToGallery,
} from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold, PiShareFatBold } from 'react-icons/pi';

export const CanvasContextMenuItems = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const saveCanvasToGallery = useSaveCanvasToGallery();
  const saveBboxToGallery = useSaveBboxToGallery();
  const saveBboxAsRegionalGuidanceIPAdapter = useSaveBboxAsRegionalGuidanceIPAdapter();
  const saveBboxAsIPAdapter = useSaveBboxAsGlobalIPAdapter();
  const saveBboxAsRasterLayer = useSaveBboxAsRasterLayer();
  const saveBboxAsControlLayer = useSaveBboxAsControlLayer();

  return (
    <>
      <MenuItem icon={<PiFloppyDiskBold />} isDisabled={isBusy} onClick={saveCanvasToGallery}>
        {t('controlLayers.saveCanvasToGallery')}
      </MenuItem>
      <MenuItem icon={<PiFloppyDiskBold />} isDisabled={isBusy} onClick={saveBboxToGallery}>
        {t('controlLayers.saveBboxToGallery')}
      </MenuItem>
      <MenuItem icon={<PiShareFatBold />} isDisabled={isBusy} onClick={saveBboxAsIPAdapter}>
        {t('controlLayers.sendBboxToGlobalIPAdapter')}
      </MenuItem>
      <MenuItem icon={<PiShareFatBold />} isDisabled={isBusy} onClick={saveBboxAsRegionalGuidanceIPAdapter}>
        {t('controlLayers.sendBboxToRegionalIPAdapter')}
      </MenuItem>
      <MenuItem icon={<PiShareFatBold />} isDisabled={isBusy} onClick={saveBboxAsControlLayer}>
        {t('controlLayers.sendBboxToControlLayer')}
      </MenuItem>
      <MenuItem icon={<PiShareFatBold />} isDisabled={isBusy} onClick={saveBboxAsRasterLayer}>
        {t('controlLayers.sendBboxToRasterLayer')}
      </MenuItem>
    </>
  );
});

CanvasContextMenuItems.displayName = 'CanvasContextMenuItems';

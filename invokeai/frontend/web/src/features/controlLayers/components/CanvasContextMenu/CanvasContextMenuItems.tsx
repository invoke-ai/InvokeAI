import { MenuItem } from '@invoke-ai/ui-library';
import {
  useIsSavingCanvas,
  useSaveBboxAsControlLayer,
  useSaveBboxAsGlobalIPAdapter,
  useSaveBboxAsRasterLayer,
  useSaveBboxAsRegionalGuidanceIPAdapter,
  useSaveBboxToGallery,
  useSaveCanvasToGallery,
} from 'features/controlLayers/hooks/saveCanvasHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold, PiShareFatBold } from 'react-icons/pi';

export const CanvasContextMenuItems = memo(() => {
  const { t } = useTranslation();
  const isSaving = useIsSavingCanvas();
  const saveCanvasToGallery = useSaveCanvasToGallery();
  const saveBboxToGallery = useSaveBboxToGallery();
  const saveBboxAsRegionalGuidanceIPAdapter = useSaveBboxAsRegionalGuidanceIPAdapter();
  const saveBboxAsIPAdapter = useSaveBboxAsGlobalIPAdapter();
  const saveBboxAsRasterLayer = useSaveBboxAsRasterLayer();
  const saveBboxAsControlLayer = useSaveBboxAsControlLayer();

  return (
    <>
      <MenuItem icon={<PiFloppyDiskBold />} isLoading={isSaving.isTrue} onClick={saveCanvasToGallery}>
        {t('controlLayers.saveCanvasToGallery')}
      </MenuItem>
      <MenuItem icon={<PiFloppyDiskBold />} isLoading={isSaving.isTrue} onClick={saveBboxToGallery}>
        {t('controlLayers.saveBboxToGallery')}
      </MenuItem>
      <MenuItem icon={<PiShareFatBold />} isLoading={isSaving.isTrue} onClick={saveBboxAsIPAdapter}>
        {t('controlLayers.sendBboxToGlobalIPAdapter')}
      </MenuItem>
      <MenuItem icon={<PiShareFatBold />} isLoading={isSaving.isTrue} onClick={saveBboxAsRegionalGuidanceIPAdapter}>
        {t('controlLayers.sendBboxToRegionalIPAdapter')}
      </MenuItem>
      <MenuItem icon={<PiShareFatBold />} isLoading={isSaving.isTrue} onClick={saveBboxAsControlLayer}>
        {t('controlLayers.sendBboxToControlLayer')}
      </MenuItem>
      <MenuItem icon={<PiShareFatBold />} isLoading={isSaving.isTrue} onClick={saveBboxAsRasterLayer}>
        {t('controlLayers.sendBboxToRasterLayer')}
      </MenuItem>
    </>
  );
});

CanvasContextMenuItems.displayName = 'CanvasContextMenuItems';

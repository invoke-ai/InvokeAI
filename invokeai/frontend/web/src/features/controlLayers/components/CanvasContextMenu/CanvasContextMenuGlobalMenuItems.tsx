import { MenuGroup, MenuItem } from '@invoke-ai/ui-library';
import { CanvasContextMenuItemsCropCanvasToBbox } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuItemsCropCanvasToBbox';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import {
  useNewControlLayerFromBbox,
  useNewGlobalReferenceImageFromBbox,
  useNewRasterLayerFromBbox,
  useNewRegionalReferenceImageFromBbox,
  useSaveBboxToGallery,
  useSaveCanvasToGallery,
} from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

export const CanvasContextMenuGlobalMenuItems = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const saveCanvasToGallery = useSaveCanvasToGallery();
  const saveBboxToGallery = useSaveBboxToGallery();
  const newRegionalReferenceImageFromBbox = useNewRegionalReferenceImageFromBbox();
  const newGlobalReferenceImageFromBbox = useNewGlobalReferenceImageFromBbox();
  const newRasterLayerFromBbox = useNewRasterLayerFromBbox();
  const newControlLayerFromBbox = useNewControlLayerFromBbox();

  return (
    <>
      <MenuGroup title={t('controlLayers.canvasContextMenu.canvasGroup')}>
        <CanvasContextMenuItemsCropCanvasToBbox />
      </MenuGroup>
      <MenuGroup title={t('controlLayers.canvasContextMenu.saveToGalleryGroup')}>
        <MenuItem icon={<PiFloppyDiskBold />} isDisabled={isBusy} onClick={saveCanvasToGallery}>
          {t('controlLayers.canvasContextMenu.saveCanvasToGallery')}
        </MenuItem>
        <MenuItem icon={<PiFloppyDiskBold />} isDisabled={isBusy} onClick={saveBboxToGallery}>
          {t('controlLayers.canvasContextMenu.saveBboxToGallery')}
        </MenuItem>
      </MenuGroup>
      <MenuGroup title={t('controlLayers.canvasContextMenu.bboxGroup')}>
        <MenuItem icon={<NewLayerIcon />} isDisabled={isBusy} onClick={newGlobalReferenceImageFromBbox}>
          {t('controlLayers.canvasContextMenu.newGlobalReferenceImage')}
        </MenuItem>
        <MenuItem icon={<NewLayerIcon />} isDisabled={isBusy} onClick={newRegionalReferenceImageFromBbox}>
          {t('controlLayers.canvasContextMenu.newRegionalReferenceImage')}
        </MenuItem>
        <MenuItem icon={<NewLayerIcon />} isDisabled={isBusy} onClick={newControlLayerFromBbox}>
          {t('controlLayers.canvasContextMenu.newControlLayer')}
        </MenuItem>
        <MenuItem icon={<NewLayerIcon />} isDisabled={isBusy} onClick={newRasterLayerFromBbox}>
          {t('controlLayers.canvasContextMenu.newRasterLayer')}
        </MenuItem>
      </MenuGroup>
    </>
  );
});

CanvasContextMenuGlobalMenuItems.displayName = 'CanvasContextMenuGlobalMenuItems';

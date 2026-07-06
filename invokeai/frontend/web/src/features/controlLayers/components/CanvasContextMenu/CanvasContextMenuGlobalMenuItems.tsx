import { Menu, MenuButton, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { CanvasContextMenuItemsCropCanvasToBbox } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuItemsCropCanvasToBbox';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import { useLoadCanvasProjectWithDialog } from 'features/controlLayers/components/LoadCanvasProjectConfirmationAlertDialog';
import { useSaveCanvasProjectWithDialog } from 'features/controlLayers/components/SaveCanvasProjectDialog';
import { useCopyCanvasToClipboard } from 'features/controlLayers/hooks/copyHooks';
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
import { PiArchiveBold, PiCopyBold, PiFileArrowDownBold, PiFileArrowUpBold, PiFloppyDiskBold } from 'react-icons/pi';

export const CanvasContextMenuGlobalMenuItems = memo(() => {
  const { t } = useTranslation();
  const saveSubMenu = useSubMenu();
  const projectSubMenu = useSubMenu();
  const newSubMenu = useSubMenu();
  const copySubMenu = useSubMenu();
  const isBusy = useCanvasIsBusy();
  const saveCanvasToGallery = useSaveCanvasToGallery();
  const saveBboxToGallery = useSaveBboxToGallery();
  const saveCanvasProject = useSaveCanvasProjectWithDialog();
  const loadCanvasProject = useLoadCanvasProjectWithDialog();
  const newRegionalReferenceImageFromBbox = useNewRegionalReferenceImageFromBbox();
  const newGlobalReferenceImageFromBbox = useNewGlobalReferenceImageFromBbox();
  const newRasterLayerFromBbox = useNewRasterLayerFromBbox();
  const newControlLayerFromBbox = useNewControlLayerFromBbox();
  const copyCanvasToClipboard = useCopyCanvasToClipboard('canvas');
  const copyBboxToClipboard = useCopyCanvasToClipboard('bbox');

  return (
    <>
      <MenuGroup title={t('controlLayers.canvasContextMenu.canvasGroup')}>
        <CanvasContextMenuItemsCropCanvasToBbox />
        <MenuItem {...saveSubMenu.parentMenuItemProps} icon={<PiFloppyDiskBold />}>
          <Menu {...saveSubMenu.menuProps}>
            <MenuButton {...saveSubMenu.menuButtonProps}>
              <SubMenuButtonContent label={t('controlLayers.canvasContextMenu.saveToGalleryGroup')} />
            </MenuButton>
            <MenuList {...saveSubMenu.menuListProps}>
              <MenuItem icon={<PiFloppyDiskBold />} isDisabled={isBusy} onClick={saveCanvasToGallery}>
                {t('controlLayers.canvasContextMenu.saveCanvasToGallery')}
              </MenuItem>
              <MenuItem icon={<PiFloppyDiskBold />} isDisabled={isBusy} onClick={saveBboxToGallery}>
                {t('controlLayers.canvasContextMenu.saveBboxToGallery')}
              </MenuItem>
            </MenuList>
          </Menu>
        </MenuItem>
        <MenuItem {...projectSubMenu.parentMenuItemProps} icon={<PiArchiveBold />}>
          <Menu {...projectSubMenu.menuProps}>
            <MenuButton {...projectSubMenu.menuButtonProps}>
              <SubMenuButtonContent label={t('controlLayers.canvasProject.project')} />
            </MenuButton>
            <MenuList {...projectSubMenu.menuListProps}>
              <MenuItem icon={<PiFileArrowDownBold />} isDisabled={isBusy} onClick={saveCanvasProject}>
                {t('controlLayers.canvasProject.saveProject')}
              </MenuItem>
              <MenuItem icon={<PiFileArrowUpBold />} isDisabled={isBusy} onClick={loadCanvasProject}>
                {t('controlLayers.canvasProject.loadProject')}
              </MenuItem>
            </MenuList>
          </Menu>
        </MenuItem>
        <MenuItem {...newSubMenu.parentMenuItemProps} icon={<NewLayerIcon />}>
          <Menu {...newSubMenu.menuProps}>
            <MenuButton {...newSubMenu.menuButtonProps}>
              <SubMenuButtonContent label={t('controlLayers.canvasContextMenu.bboxGroup')} />
            </MenuButton>
            <MenuList {...newSubMenu.menuListProps}>
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
            </MenuList>
          </Menu>
        </MenuItem>
        <MenuItem {...copySubMenu.parentMenuItemProps} icon={<PiCopyBold />}>
          <Menu {...copySubMenu.menuProps}>
            <MenuButton {...copySubMenu.menuButtonProps}>
              <SubMenuButtonContent label={t('controlLayers.canvasContextMenu.copyToClipboard')} />
            </MenuButton>
            <MenuList {...copySubMenu.menuListProps}>
              <MenuItem icon={<PiCopyBold />} isDisabled={isBusy} onClick={copyCanvasToClipboard}>
                {t('controlLayers.canvasContextMenu.copyCanvasToClipboard')}
              </MenuItem>
              <MenuItem icon={<PiCopyBold />} isDisabled={isBusy} onClick={copyBboxToClipboard}>
                {t('controlLayers.canvasContextMenu.copyBboxToClipboard')}
              </MenuItem>
            </MenuList>
          </Menu>
        </MenuItem>
      </MenuGroup>
    </>
  );
});

CanvasContextMenuGlobalMenuItems.displayName = 'CanvasContextMenuGlobalMenuItems';

import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useCanvasIsBusySafe } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useItemDTOContext, useItemDTOContextImageOnly } from 'features/gallery/contexts/ItemDTOContext';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileBold, PiPlusBold } from 'react-icons/pi';
import { isImageDTO } from 'services/api/types';

export const ContextMenuItemNewCanvasFromImageSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const store = useAppStore();
  const imageDTO = useItemDTOContextImageOnly();
  const isBusy = useCanvasIsBusySafe();
  const isStaging = useCanvasIsStaging();

  const onClickNewCanvasWithRasterLayerFromImage = useCallback(async () => {
    const { dispatch, getState } = store;
    await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
    await newCanvasFromImage({
      imageDTO,
      withResize: false,
      withInpaintMask: true,
      type: 'raster_layer',
      dispatch,
      getState,
    });
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, store, t]);

  const onClickNewCanvasWithControlLayerFromImage = useCallback(async () => {
    const { dispatch, getState } = store;
    await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
    await newCanvasFromImage({
      imageDTO,
      withResize: false,
      withInpaintMask: true,
      type: 'control_layer',
      dispatch,
      getState,
    });
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, store, t]);

  const onClickNewCanvasWithRasterLayerFromImageWithResize = useCallback(async () => {
    const { dispatch, getState } = store;
    await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
    await newCanvasFromImage({
      imageDTO,
      withResize: true,
      withInpaintMask: true,
      type: 'raster_layer',
      dispatch,
      getState,
    });
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
        status: 'success',
      });
  }, [imageDTO, store, t]);

  const onClickNewCanvasWithControlLayerFromImageWithResize = useCallback(async () => {
    const { dispatch, getState } = store;
    await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
    await newCanvasFromImage({
      imageDTO,
      withResize: true,
      withInpaintMask: true,
      type: 'control_layer',
      dispatch,
      getState,
    });
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
        status: 'success',
      });
  }, [imageDTO, store, t]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiPlusBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.newCanvasFromImage')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem
            icon={<PiFileBold />}
            onClickCapture={onClickNewCanvasWithRasterLayerFromImage}
            isDisabled={isStaging || isBusy}
          >
            {t('controlLayers.asRasterLayer')}
          </MenuItem>
          <MenuItem
            icon={<PiFileBold />}
            onClickCapture={onClickNewCanvasWithRasterLayerFromImageWithResize}
            isDisabled={isStaging || isBusy}
          >
            {t('controlLayers.asRasterLayerResize')}
          </MenuItem>
          <MenuItem
            icon={<PiFileBold />}
            onClickCapture={onClickNewCanvasWithControlLayerFromImage}
            isDisabled={isStaging || isBusy}
          >
            {t('controlLayers.asControlLayer')}
          </MenuItem>
          <MenuItem
            icon={<PiFileBold />}
            onClickCapture={onClickNewCanvasWithControlLayerFromImageWithResize}
            isDisabled={isStaging || isBusy}
          >
            {t('controlLayers.asControlLayerResize')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ContextMenuItemNewCanvasFromImageSubMenu.displayName = 'ContextMenuItemNewCanvasFromImageSubMenu';

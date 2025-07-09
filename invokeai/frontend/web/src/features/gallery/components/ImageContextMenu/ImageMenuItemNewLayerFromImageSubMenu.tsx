import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import { useCanvasIsBusySafe } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { sentImageToCanvas } from 'features/gallery/store/actions';
import { createNewCanvasEntityFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const ImageMenuItemNewLayerFromImageSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const store = useAppStore();
  const imageDTO = useImageDTOContext();
  const isBusy = useCanvasIsBusySafe();
  const activeTab = useAppSelector(selectActiveTab);

  const onClickNewRasterLayerFromImage = useCallback(async () => {
    const { dispatch, getState } = store;
    await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
    createNewCanvasEntityFromImage({ imageDTO, type: 'raster_layer', dispatch, getState });
    dispatch(sentImageToCanvas());
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, store, t]);

  const onClickNewControlLayerFromImage = useCallback(async () => {
    const { dispatch, getState } = store;
    await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
    createNewCanvasEntityFromImage({ imageDTO, type: 'control_layer', dispatch, getState });
    dispatch(sentImageToCanvas());
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, store, t]);

  const onClickNewInpaintMaskFromImage = useCallback(async () => {
    const { dispatch, getState } = store;
    await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
    createNewCanvasEntityFromImage({ imageDTO, type: 'inpaint_mask', dispatch, getState });
    dispatch(sentImageToCanvas());
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, store, t]);

  const onClickNewRegionalGuidanceFromImage = useCallback(async () => {
    const { dispatch, getState } = store;
    await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
    createNewCanvasEntityFromImage({ imageDTO, type: 'regional_guidance', dispatch, getState });
    dispatch(sentImageToCanvas());
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, store, t]);

  const onClickNewRegionalReferenceImageFromImage = useCallback(async () => {
    const { dispatch, getState } = store;
    await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
    createNewCanvasEntityFromImage({ imageDTO, type: 'regional_guidance_with_reference_image', dispatch, getState });
    dispatch(sentImageToCanvas());
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, store, t]);

  if (activeTab === 'generate') {
    return null;
  }

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiPlusBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.newLayerFromImage')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem icon={<NewLayerIcon />} onClickCapture={onClickNewInpaintMaskFromImage} isDisabled={isBusy}>
            {t('controlLayers.inpaintMask')}
          </MenuItem>
          <MenuItem icon={<NewLayerIcon />} onClickCapture={onClickNewRegionalGuidanceFromImage} isDisabled={isBusy}>
            {t('controlLayers.regionalGuidance')}
          </MenuItem>
          <MenuItem icon={<NewLayerIcon />} onClickCapture={onClickNewControlLayerFromImage} isDisabled={isBusy}>
            {t('controlLayers.controlLayer')}
          </MenuItem>
          <MenuItem icon={<NewLayerIcon />} onClickCapture={onClickNewRasterLayerFromImage} isDisabled={isBusy}>
            {t('controlLayers.rasterLayer')}
          </MenuItem>
          <MenuItem
            icon={<NewLayerIcon />}
            onClickCapture={onClickNewRegionalReferenceImageFromImage}
            isDisabled={isBusy}
          >
            {t('controlLayers.referenceImageRegional')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ImageMenuItemNewLayerFromImageSubMenu.displayName = 'ImageMenuItemNewLayerFromImageSubMenu';

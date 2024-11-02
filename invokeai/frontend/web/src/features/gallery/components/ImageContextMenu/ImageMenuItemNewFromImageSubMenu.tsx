import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { sentImageToCanvas } from 'features/gallery/store/actions';
import {
  newCanvasEntityFromImageActionApi,
  newCanvasFromImageActionApi,
  singleImageSourceApi,
} from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileBold, PiPlusBold } from 'react-icons/pi';

export const ImageMenuItemNewFromImageSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const store = useAppStore();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();
  const isBusy = useCanvasIsBusy();

  const onClickNewCanvasWithRasterLayerFromImage = useCallback(() => {
    const targetData = newCanvasFromImageActionApi.getData({ type: 'raster_layer' });
    const sourceData = singleImageSourceApi.getData({ imageDTO });
    newCanvasFromImageActionApi.handler(sourceData, targetData, store.dispatch, store.getState);
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, store.dispatch, store.getState, t]);

  const onClickNewCanvasWithControlLayerFromImage = useCallback(() => {
    const targetData = newCanvasFromImageActionApi.getData({ type: 'control_layer' });
    const sourceData = singleImageSourceApi.getData({ imageDTO });
    newCanvasFromImageActionApi.handler(sourceData, targetData, store.dispatch, store.getState);
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, store.dispatch, store.getState, t]);

  const onClickNewRasterLayerFromImage = useCallback(() => {
    const targetData = newCanvasEntityFromImageActionApi.getData({ type: 'raster_layer' });
    const sourceData = singleImageSourceApi.getData({ imageDTO });
    newCanvasEntityFromImageActionApi.handler(sourceData, targetData, store.dispatch, store.getState);
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, store.dispatch, store.getState, t]);

  const onClickNewControlLayerFromImage = useCallback(() => {
    const targetData = newCanvasEntityFromImageActionApi.getData({ type: 'control_layer' });
    const sourceData = singleImageSourceApi.getData({ imageDTO });
    newCanvasEntityFromImageActionApi.handler(sourceData, targetData, store.dispatch, store.getState);
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, store.dispatch, store.getState, t]);

  const onClickNewInpaintMaskFromImage = useCallback(() => {
    const targetData = newCanvasEntityFromImageActionApi.getData({ type: 'inpaint_mask' });
    const sourceData = singleImageSourceApi.getData({ imageDTO });
    newCanvasEntityFromImageActionApi.handler(sourceData, targetData, store.dispatch, store.getState);
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, store.dispatch, store.getState, t]);

  const onClickNewRegionalGuidanceFromImage = useCallback(() => {
    const targetData = newCanvasEntityFromImageActionApi.getData({ type: 'regional_guidance' });
    const sourceData = singleImageSourceApi.getData({ imageDTO });
    newCanvasEntityFromImageActionApi.handler(sourceData, targetData, store.dispatch, store.getState);
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, store.dispatch, store.getState, t]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiPlusBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('controlLayers.newFromImage')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem icon={<PiFileBold />} onClickCapture={onClickNewCanvasWithRasterLayerFromImage} isDisabled={isBusy}>
            {t('controlLayers.canvasAsRasterLayer')}
          </MenuItem>
          <MenuItem
            icon={<PiFileBold />}
            onClickCapture={onClickNewCanvasWithControlLayerFromImage}
            isDisabled={isBusy}
          >
            {t('controlLayers.canvasAsControlLayer')}
          </MenuItem>
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
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ImageMenuItemNewFromImageSubMenu.displayName = 'ImageMenuItemNewFromImageSubMenu';

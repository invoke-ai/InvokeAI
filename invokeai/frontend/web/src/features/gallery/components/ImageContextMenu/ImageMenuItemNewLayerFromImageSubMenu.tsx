import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { selectIsFLUX, selectIsSD3 } from 'features/controlLayers/store/paramsSlice';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { sentImageToCanvas } from 'features/gallery/store/actions';
import { createNewCanvasEntityFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const ImageMenuItemNewLayerFromImageSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const store = useAppStore();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();
  const isBusy = useCanvasIsBusy();
  const isFLUX = useAppSelector(selectIsFLUX);
  const isSD3 = useAppSelector(selectIsSD3);

  const onClickNewRasterLayerFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    createNewCanvasEntityFromImage({ imageDTO, type: 'raster_layer', dispatch, getState });
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

  const onClickNewControlLayerFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    createNewCanvasEntityFromImage({ imageDTO, type: 'control_layer', dispatch, getState });
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

  const onClickNewInpaintMaskFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    createNewCanvasEntityFromImage({ imageDTO, type: 'inpaint_mask', dispatch, getState });
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

  const onClickNewRegionalGuidanceFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    createNewCanvasEntityFromImage({ imageDTO, type: 'regional_guidance', dispatch, getState });
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

  const onClickNewGlobalReferenceImageFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    createNewCanvasEntityFromImage({ imageDTO, type: 'reference_image', dispatch, getState });
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

  const onClickNewRegionalReferenceImageFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    createNewCanvasEntityFromImage({ imageDTO, type: 'regional_guidance_with_reference_image', dispatch, getState });
    dispatch(sentImageToCanvas());
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

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
          <MenuItem
            icon={<NewLayerIcon />}
            onClickCapture={onClickNewRegionalGuidanceFromImage}
            isDisabled={isBusy || isFLUX || isSD3}
          >
            {t('controlLayers.regionalGuidance')}
          </MenuItem>
          <MenuItem
            icon={<NewLayerIcon />}
            onClickCapture={onClickNewControlLayerFromImage}
            isDisabled={isBusy || isSD3}
          >
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
          <MenuItem
            icon={<NewLayerIcon />}
            onClickCapture={onClickNewGlobalReferenceImageFromImage}
            isDisabled={isBusy}
          >
            {t('controlLayers.referenceImageGlobal')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ImageMenuItemNewLayerFromImageSubMenu.displayName = 'ImageMenuItemNewLayerFromImageSubMenu';

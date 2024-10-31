import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import {
  useNewCanvasFromImage,
  useNewControlLayerFromImage,
  useNewInpaintMaskFromImage,
  useNewRasterLayerFromImage,
  useNewRegionalGuidanceFromImage,
} from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { sentImageToCanvas } from 'features/gallery/store/actions';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileBold, PiPlusBold } from 'react-icons/pi';

export const ImageMenuItemNewFromImageSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const dispatch = useAppDispatch();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();
  const isBusy = useCanvasIsBusy();
  const newRasterLayerFromImage = useNewRasterLayerFromImage();
  const newControlLayerFromImage = useNewControlLayerFromImage();
  const newInpaintMaskFromImage = useNewInpaintMaskFromImage();
  const newRegionalGuidanceFromImage = useNewRegionalGuidanceFromImage();
  const newCanvasFromImage = useNewCanvasFromImage();

  const onClickNewCanvasWithRasterLayerFromImage = useCallback(() => {
    newCanvasFromImage(imageDTO, 'raster_layer');
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, newCanvasFromImage, t]);

  const onClickNewCanvasWithControlLayerFromImage = useCallback(() => {
    newCanvasFromImage(imageDTO, 'control_layer');
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, newCanvasFromImage, t]);

  const onClickNewRasterLayerFromImage = useCallback(() => {
    dispatch(sentImageToCanvas());
    newRasterLayerFromImage(imageDTO);
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, newRasterLayerFromImage, t]);

  const onClickNewControlLayerFromImage = useCallback(() => {
    dispatch(sentImageToCanvas());
    newControlLayerFromImage(imageDTO);
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, newControlLayerFromImage, t]);

  const onClickNewInpaintMaskFromImage = useCallback(() => {
    dispatch(sentImageToCanvas());
    newInpaintMaskFromImage(imageDTO);
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, newInpaintMaskFromImage, t]);

  const onClickNewRegionalGuidanceFromImage = useCallback(() => {
    dispatch(sentImageToCanvas());
    newRegionalGuidanceFromImage(imageDTO);
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, imageViewer, newRegionalGuidanceFromImage, t]);

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

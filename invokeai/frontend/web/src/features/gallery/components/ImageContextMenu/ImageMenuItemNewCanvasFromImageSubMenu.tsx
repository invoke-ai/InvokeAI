import { Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { selectIsSD3 } from 'features/controlLayers/store/paramsSlice';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileBold, PiPlusBold } from 'react-icons/pi';

export const ImageMenuItemNewCanvasFromImageSubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const store = useAppStore();
  const imageDTO = useImageDTOContext();
  const imageViewer = useImageViewer();
  const isBusy = useCanvasIsBusy();
  const isSD3 = useAppSelector(selectIsSD3);

  const onClickNewCanvasWithRasterLayerFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    newCanvasFromImage({ imageDTO, withResize: false, type: 'raster_layer', dispatch, getState });
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

  const onClickNewCanvasWithControlLayerFromImage = useCallback(() => {
    const { dispatch, getState } = store;
    newCanvasFromImage({ imageDTO, withResize: false, type: 'control_layer', dispatch, getState });
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

  const onClickNewCanvasWithRasterLayerFromImageWithResize = useCallback(() => {
    const { dispatch, getState } = store;
    newCanvasFromImage({ imageDTO, withResize: true, type: 'raster_layer', dispatch, getState });
    dispatch(setActiveTab('canvas'));
    imageViewer.close();
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, imageViewer, store, t]);

  const onClickNewCanvasWithControlLayerFromImageWithResize = useCallback(() => {
    const { dispatch, getState } = store;
    newCanvasFromImage({ imageDTO, withResize: true, type: 'control_layer', dispatch, getState });
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
          <SubMenuButtonContent label={t('controlLayers.newCanvasFromImage')} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem icon={<PiFileBold />} onClickCapture={onClickNewCanvasWithRasterLayerFromImage} isDisabled={isBusy}>
            {t('controlLayers.asRasterLayer')}
          </MenuItem>
          <MenuItem
            icon={<PiFileBold />}
            onClickCapture={onClickNewCanvasWithRasterLayerFromImageWithResize}
            isDisabled={isBusy}
          >
            {t('controlLayers.asRasterLayerResize')}
          </MenuItem>
          <MenuItem
            icon={<PiFileBold />}
            onClickCapture={onClickNewCanvasWithControlLayerFromImage}
            isDisabled={isBusy || isSD3}
          >
            {t('controlLayers.asControlLayer')}
          </MenuItem>
          <MenuItem
            icon={<PiFileBold />}
            onClickCapture={onClickNewCanvasWithControlLayerFromImageWithResize}
            isDisabled={isBusy || isSD3}
          >
            {t('controlLayers.asControlLayerResize')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});

ImageMenuItemNewCanvasFromImageSubMenu.displayName = 'ImageMenuItemNewCanvasFromImageSubMenu';

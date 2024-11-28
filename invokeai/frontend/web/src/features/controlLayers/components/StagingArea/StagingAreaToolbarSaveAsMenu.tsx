import { IconButton, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import { selectSelectedImage } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { createNewCanvasEntityFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotsThreeBold } from 'react-icons/pi';
import { imageDTOToFile, uploadImage } from 'services/api/endpoints/images';

const uploadImageArg = { image_category: 'general', is_intermediate: true, silent: true } as const;

export const StagingAreaToolbarSaveAsMenu = memo(() => {
  const { t } = useTranslation();
  const selectedImage = useAppSelector(selectSelectedImage);
  const store = useAppStore();

  const onClickNewRasterLayerFromImage = useCallback(async () => {
    if (!selectedImage) {
      return;
    }
    const { dispatch, getState } = store;
    const file = await imageDTOToFile(selectedImage.imageDTO);
    const imageDTO = await uploadImage({ file, ...uploadImageArg });
    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'raster_layer',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [selectedImage, store, t]);

  const onClickNewControlLayerFromImage = useCallback(async () => {
    if (!selectedImage) {
      return;
    }
    const { dispatch, getState } = store;
    const file = await imageDTOToFile(selectedImage.imageDTO);
    const imageDTO = await uploadImage({ file, ...uploadImageArg });
    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'control_layer',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [selectedImage, store, t]);

  const onClickNewInpaintMaskFromImage = useCallback(async () => {
    if (!selectedImage) {
      return;
    }
    const { dispatch, getState } = store;
    const file = await imageDTOToFile(selectedImage.imageDTO);
    const imageDTO = await uploadImage({ file, ...uploadImageArg });
    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'inpaint_mask',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [selectedImage, store, t]);

  const onClickNewRegionalGuidanceFromImage = useCallback(async () => {
    if (!selectedImage) {
      return;
    }
    const { dispatch, getState } = store;
    const file = await imageDTOToFile(selectedImage.imageDTO);
    const imageDTO = await uploadImage({ file, ...uploadImageArg });
    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'regional_guidance',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [selectedImage, store, t]);

  return (
    <Menu>
      <MenuButton
        as={IconButton}
        aria-label={t('controlLayers.newLayerFromImage')}
        tooltip={t('controlLayers.newLayerFromImage')}
        icon={<PiDotsThreeBold />}
        colorScheme="invokeBlue"
        isDisabled={!selectedImage}
      />
      <MenuList>
        <MenuItem icon={<NewLayerIcon />} onClickCapture={onClickNewInpaintMaskFromImage} isDisabled={!selectedImage}>
          {t('controlLayers.inpaintMask')}
        </MenuItem>
        <MenuItem
          icon={<NewLayerIcon />}
          onClickCapture={onClickNewRegionalGuidanceFromImage}
          isDisabled={!selectedImage}
        >
          {t('controlLayers.regionalGuidance')}
        </MenuItem>
        <MenuItem icon={<NewLayerIcon />} onClickCapture={onClickNewControlLayerFromImage} isDisabled={!selectedImage}>
          {t('controlLayers.controlLayer')}
        </MenuItem>
        <MenuItem icon={<NewLayerIcon />} onClickCapture={onClickNewRasterLayerFromImage} isDisabled={!selectedImage}>
          {t('controlLayers.rasterLayer')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

StagingAreaToolbarSaveAsMenu.displayName = 'StagingAreaToolbarSaveAsMenu';

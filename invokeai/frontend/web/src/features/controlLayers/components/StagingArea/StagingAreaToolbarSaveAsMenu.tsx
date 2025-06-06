import { IconButton, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppStore } from 'app/store/nanostores/store';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { createNewCanvasEntityFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotsThreeBold } from 'react-icons/pi';
import { copyImage } from 'services/api/endpoints/images';

const uploadImageArg = { image_category: 'general', is_intermediate: true, silent: true } as const;

export const StagingAreaToolbarSaveAsMenu = memo(() => {
  const canvasManager = useCanvasManager();
  const { t } = useTranslation();
  const ctx = useCanvasSessionContext();
  const imageName = useStore(ctx.$selectedItemOutputImageName);
  const store = useAppStore();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  const toastSentToCanvas = useCallback(() => {
    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [t]);

  const onClickNewRasterLayerFromImage = useCallback(async () => {
    if (!imageName) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(imageName, uploadImageArg);
    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'raster_layer',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [imageName, store, toastSentToCanvas]);

  const onClickNewControlLayerFromImage = useCallback(async () => {
    if (!imageName) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(imageName, uploadImageArg);

    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'control_layer',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [imageName, store, toastSentToCanvas]);

  const onClickNewInpaintMaskFromImage = useCallback(async () => {
    if (!imageName) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(imageName, uploadImageArg);

    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'inpaint_mask',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [imageName, store, toastSentToCanvas]);

  const onClickNewRegionalGuidanceFromImage = useCallback(async () => {
    if (!imageName) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(imageName, uploadImageArg);

    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'regional_guidance',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [imageName, store, toastSentToCanvas]);

  return (
    <Menu>
      <MenuButton
        as={IconButton}
        aria-label={t('controlLayers.newLayerFromImage')}
        tooltip={t('controlLayers.newLayerFromImage')}
        icon={<PiDotsThreeBold />}
        colorScheme="invokeBlue"
        isDisabled={!imageName || !shouldShowStagedImage}
      />
      <MenuList>
        <MenuItem
          icon={<NewLayerIcon />}
          onClickCapture={onClickNewInpaintMaskFromImage}
          isDisabled={!imageName || !shouldShowStagedImage}
        >
          {t('controlLayers.inpaintMask')}
        </MenuItem>
        <MenuItem
          icon={<NewLayerIcon />}
          onClickCapture={onClickNewRegionalGuidanceFromImage}
          isDisabled={!imageName || !shouldShowStagedImage}
        >
          {t('controlLayers.regionalGuidance')}
        </MenuItem>
        <MenuItem
          icon={<NewLayerIcon />}
          onClickCapture={onClickNewControlLayerFromImage}
          isDisabled={!imageName || !shouldShowStagedImage}
        >
          {t('controlLayers.controlLayer')}
        </MenuItem>
        <MenuItem
          icon={<NewLayerIcon />}
          onClickCapture={onClickNewRasterLayerFromImage}
          isDisabled={!imageName || !shouldShowStagedImage}
        >
          {t('controlLayers.rasterLayer')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

StagingAreaToolbarSaveAsMenu.displayName = 'StagingAreaToolbarSaveAsMenu';

import { MenuGroup, MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppStore } from 'app/store/nanostores/store';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { createNewCanvasEntityFromImage } from 'features/imageActions/actions';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { copyImage } from 'services/api/endpoints/images';

const uploadImageArg = { image_category: 'general', is_intermediate: true, silent: true } as const;

export const StagingAreaToolbarNewLayerFromImageMenuItems = memo(() => {
  const canvasManager = useCanvasManager();
  const { t } = useTranslation();
  const ctx = useCanvasSessionContext();
  const selectedItemOutputImageDTO = useStore(ctx.$selectedItemOutputImageDTO);
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
    if (!selectedItemOutputImageDTO) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(selectedItemOutputImageDTO.image_name, uploadImageArg);
    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'raster_layer',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [selectedItemOutputImageDTO, store, toastSentToCanvas]);

  const onClickNewControlLayerFromImage = useCallback(async () => {
    if (!selectedItemOutputImageDTO) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(selectedItemOutputImageDTO.image_name, uploadImageArg);

    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'control_layer',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [selectedItemOutputImageDTO, store, toastSentToCanvas]);

  const onClickNewInpaintMaskFromImage = useCallback(async () => {
    if (!selectedItemOutputImageDTO) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(selectedItemOutputImageDTO.image_name, uploadImageArg);

    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'inpaint_mask',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [selectedItemOutputImageDTO, store, toastSentToCanvas]);

  const onClickNewRegionalGuidanceFromImage = useCallback(async () => {
    if (!selectedItemOutputImageDTO) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(selectedItemOutputImageDTO.image_name, uploadImageArg);

    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'regional_guidance',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [selectedItemOutputImageDTO, store, toastSentToCanvas]);

  return (
    <MenuGroup title="New Layer From Image">
      <MenuItem
        icon={<NewLayerIcon />}
        onClickCapture={onClickNewInpaintMaskFromImage}
        isDisabled={!selectedItemOutputImageDTO || !shouldShowStagedImage}
      >
        {t('controlLayers.inpaintMask')}
      </MenuItem>
      <MenuItem
        icon={<NewLayerIcon />}
        onClickCapture={onClickNewRegionalGuidanceFromImage}
        isDisabled={!selectedItemOutputImageDTO || !shouldShowStagedImage}
      >
        {t('controlLayers.regionalGuidance')}
      </MenuItem>
      <MenuItem
        icon={<NewLayerIcon />}
        onClickCapture={onClickNewControlLayerFromImage}
        isDisabled={!selectedItemOutputImageDTO || !shouldShowStagedImage}
      >
        {t('controlLayers.controlLayer')}
      </MenuItem>
      <MenuItem
        icon={<NewLayerIcon />}
        onClickCapture={onClickNewRasterLayerFromImage}
        isDisabled={!selectedItemOutputImageDTO || !shouldShowStagedImage}
      >
        {t('controlLayers.rasterLayer')}
      </MenuItem>
    </MenuGroup>
  );
});

StagingAreaToolbarNewLayerFromImageMenuItems.displayName = 'StagingAreaToolbarNewLayerFromImageMenuItems';

import { MenuGroup, MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppStore } from 'app/store/storeHooks';
import { NewLayerIcon } from 'features/controlLayers/components/common/icons';
import { useStagingAreaContext } from 'features/controlLayers/components/SimpleSession/context';
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
  const ctx = useStagingAreaContext();
  const selectedItemImageDTO = useStore(ctx.$selectedItemImageDTO);
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
    if (!selectedItemImageDTO) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(selectedItemImageDTO.image_name, uploadImageArg);
    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'raster_layer',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [selectedItemImageDTO, store, toastSentToCanvas]);

  const onClickNewControlLayerFromImage = useCallback(async () => {
    if (!selectedItemImageDTO) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(selectedItemImageDTO.image_name, uploadImageArg);

    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'control_layer',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [selectedItemImageDTO, store, toastSentToCanvas]);

  const onClickNewInpaintMaskFromImage = useCallback(async () => {
    if (!selectedItemImageDTO) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(selectedItemImageDTO.image_name, uploadImageArg);

    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'inpaint_mask',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [selectedItemImageDTO, store, toastSentToCanvas]);

  const onClickNewRegionalGuidanceFromImage = useCallback(async () => {
    if (!selectedItemImageDTO) {
      return;
    }
    const { dispatch, getState } = store;
    const imageDTO = await copyImage(selectedItemImageDTO.image_name, uploadImageArg);

    createNewCanvasEntityFromImage({
      imageDTO,
      type: 'regional_guidance',
      dispatch,
      getState,
      overrides: { isEnabled: false }, // We are adding the layer while staging, it should be disabled by default
    });
    toastSentToCanvas();
  }, [selectedItemImageDTO, store, toastSentToCanvas]);

  return (
    <MenuGroup title="New Layer From Image">
      <MenuItem
        icon={<NewLayerIcon />}
        onClickCapture={onClickNewInpaintMaskFromImage}
        isDisabled={!selectedItemImageDTO || !shouldShowStagedImage}
      >
        {t('controlLayers.inpaintMask')}
      </MenuItem>
      <MenuItem
        icon={<NewLayerIcon />}
        onClickCapture={onClickNewRegionalGuidanceFromImage}
        isDisabled={!selectedItemImageDTO || !shouldShowStagedImage}
      >
        {t('controlLayers.regionalGuidance')}
      </MenuItem>
      <MenuItem
        icon={<NewLayerIcon />}
        onClickCapture={onClickNewControlLayerFromImage}
        isDisabled={!selectedItemImageDTO || !shouldShowStagedImage}
      >
        {t('controlLayers.controlLayer')}
      </MenuItem>
      <MenuItem
        icon={<NewLayerIcon />}
        onClickCapture={onClickNewRasterLayerFromImage}
        isDisabled={!selectedItemImageDTO || !shouldShowStagedImage}
      >
        {t('controlLayers.rasterLayer')}
      </MenuItem>
    </MenuGroup>
  );
});

StagingAreaToolbarNewLayerFromImageMenuItems.displayName = 'StagingAreaToolbarNewLayerFromImageMenuItems';

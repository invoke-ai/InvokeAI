import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import {
  selectImageCount,
  selectSelectedImage,
  stagingAreaReset,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectBboxRect, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';

export const StagingAreaToolbarAcceptButton = memo(() => {
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const bboxRect = useAppSelector(selectBboxRect);
  const selectedImage = useAppSelector(selectSelectedImage);
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const imageCount = useAppSelector(selectImageCount);
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);
  const isCanvasFocused = useIsRegionFocused('canvas');

  const { t } = useTranslation();

  const acceptSelected = useCallback(() => {
    if (!selectedImage) {
      return;
    }
    const { x, y } = bboxRect;
    const { imageDTO, offsetX, offsetY } = selectedImage;
    const imageObject = imageDTOToImageObject(imageDTO);
    const overrides: Partial<CanvasRasterLayerState> = {
      position: { x: x + offsetX, y: y + offsetY },
      objects: [imageObject],
    };

    dispatch(rasterLayerAdded({ overrides, isSelected: selectedEntityIdentifier?.type === 'raster_layer' }));
    dispatch(stagingAreaReset());
  }, [bboxRect, dispatch, selectedEntityIdentifier?.type, selectedImage]);

  useHotkeys(
    ['enter'],
    acceptSelected,
    {
      preventDefault: true,
      enabled: isCanvasFocused && shouldShowStagedImage && imageCount > 1,
    },
    [isCanvasFocused, shouldShowStagedImage, imageCount]
  );

  return (
    <IconButton
      tooltip={`${t('common.accept')} (Enter)`}
      aria-label={`${t('common.accept')} (Enter)`}
      icon={<PiCheckBold />}
      onClick={acceptSelected}
      colorScheme="invokeBlue"
      isDisabled={!selectedImage}
    />
  );
});

StagingAreaToolbarAcceptButton.displayName = 'StagingAreaToolbarAcceptButton';

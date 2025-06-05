import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { selectImageCount, stagingAreaReset } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectBboxRect, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageNameToImageObject } from 'features/controlLayers/store/util';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';

export const StagingAreaToolbarAcceptButton = memo(() => {
  const ctx = useCanvasSessionContext();
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const bboxRect = useAppSelector(selectBboxRect);
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const imageCount = useAppSelector(selectImageCount);
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);
  const isCanvasFocused = useIsRegionFocused('canvas');
  const selectedItemImageName = useStore(ctx.$selectedItemOutputImageName);

  const { t } = useTranslation();

  const acceptSelected = useCallback(() => {
    if (!selectedItemImageName) {
      return;
    }
    const { x, y, width, height } = bboxRect;
    const imageObject = imageNameToImageObject(selectedItemImageName, { width, height });
    const overrides: Partial<CanvasRasterLayerState> = {
      position: { x, y },
      objects: [imageObject],
    };

    dispatch(rasterLayerAdded({ overrides, isSelected: selectedEntityIdentifier?.type === 'raster_layer' }));
    dispatch(stagingAreaReset());
  }, [bboxRect, selectedItemImageName, dispatch, selectedEntityIdentifier?.type]);

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
      isDisabled={!selectedItemImageName}
    />
  );
});

StagingAreaToolbarAcceptButton.displayName = 'StagingAreaToolbarAcceptButton';

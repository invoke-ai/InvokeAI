import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { canvasSessionGenerationFinished } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectBboxRect, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageNameToImageObject } from 'features/controlLayers/store/util';
import { useDeleteQueueItemsByDestination } from 'features/queue/hooks/useDeleteQueueItemsByDestination';
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
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);
  const isCanvasFocused = useIsRegionFocused('canvas');
  const selectedItemImageName = useStore(ctx.$selectedItemOutputImageName);
  const deleteQueueItemsByDestination = useDeleteQueueItemsByDestination();

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
    dispatch(canvasSessionGenerationFinished());
    deleteQueueItemsByDestination.trigger(ctx.session.id);
  }, [
    selectedItemImageName,
    bboxRect,
    dispatch,
    selectedEntityIdentifier?.type,
    deleteQueueItemsByDestination,
    ctx.session.id,
  ]);

  useHotkeys(
    ['enter'],
    acceptSelected,
    {
      preventDefault: true,
      enabled: isCanvasFocused && shouldShowStagedImage && selectedItemImageName !== null,
    },
    [isCanvasFocused, shouldShowStagedImage, selectedItemImageName]
  );

  return (
    <IconButton
      tooltip={`${t('common.accept')} (Enter)`}
      aria-label={`${t('common.accept')} (Enter)`}
      icon={<PiCheckBold />}
      onClick={acceptSelected}
      colorScheme="invokeBlue"
      isDisabled={!selectedItemImageName || !shouldShowStagedImage || deleteQueueItemsByDestination.isDisabled}
      isLoading={deleteQueueItemsByDestination.isLoading}
    />
  );
});

StagingAreaToolbarAcceptButton.displayName = 'StagingAreaToolbarAcceptButton';

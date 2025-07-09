import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { canvasSessionReset } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectBboxRect, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageNameToImageObject } from 'features/controlLayers/store/util';
import { useCancelQueueItemsByDestination } from 'features/queue/hooks/useCancelQueueItemsByDestination';
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
  const selectedItemImageDTO = useStore(ctx.$selectedItemOutputImageDTO);
  const cancelQueueItemsByDestination = useCancelQueueItemsByDestination();

  const { t } = useTranslation();

  const acceptSelected = useCallback(() => {
    if (!selectedItemImageDTO) {
      return;
    }
    const { x, y, width, height } = bboxRect;
    const imageObject = imageNameToImageObject(selectedItemImageDTO.image_name, { width, height });
    const overrides: Partial<CanvasRasterLayerState> = {
      position: { x, y },
      objects: [imageObject],
    };

    dispatch(rasterLayerAdded({ overrides, isSelected: selectedEntityIdentifier?.type === 'raster_layer' }));
    dispatch(canvasSessionReset());
    cancelQueueItemsByDestination.trigger(ctx.session.id, { withToast: false });
  }, [
    selectedItemImageDTO,
    bboxRect,
    dispatch,
    selectedEntityIdentifier?.type,
    cancelQueueItemsByDestination,
    ctx.session.id,
  ]);

  useHotkeys(
    ['enter'],
    acceptSelected,
    {
      preventDefault: true,
      enabled: isCanvasFocused && shouldShowStagedImage && selectedItemImageDTO !== null,
    },
    [isCanvasFocused, shouldShowStagedImage, selectedItemImageDTO]
  );

  return (
    <IconButton
      tooltip={`${t('common.accept')} (Enter)`}
      aria-label={`${t('common.accept')} (Enter)`}
      icon={<PiCheckBold />}
      onClick={acceptSelected}
      colorScheme="invokeBlue"
      isDisabled={!selectedItemImageDTO || !shouldShowStagedImage || cancelQueueItemsByDestination.isDisabled}
      isLoading={cancelQueueItemsByDestination.isLoading}
    />
  );
});

StagingAreaToolbarAcceptButton.displayName = 'StagingAreaToolbarAcceptButton';

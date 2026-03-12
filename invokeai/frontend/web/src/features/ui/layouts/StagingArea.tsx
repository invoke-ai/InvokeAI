import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useStagingAreaContext } from 'features/controlLayers/components/StagingArea/context';
import { STAGING_AREA_THUMBNAIL_STRIP_HEIGHT } from 'features/controlLayers/components/StagingArea/shared';
import { StagingAreaItemsList } from 'features/controlLayers/components/StagingArea/StagingAreaItemsList';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  canvasSessionThumbnailsVisibilityToggled,
  selectCanvasSessionAreThumbnailsVisible,
  useCanvasIsStaging,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback, useEffect } from 'react';

export const StagingArea = memo(() => {
  const dispatch = useAppDispatch();
  const isStaging = useCanvasIsStaging();
  const canvasManager = useCanvasManager();
  const ctx = useStagingAreaContext();
  const areThumbnailsVisible = useAppSelector(selectCanvasSessionAreThumbnailsVisible);

  useEffect(() => {
    return canvasManager.stagingArea.connectToSession(ctx.$items, ctx.$selectedItem);
  }, [canvasManager, ctx.$items, ctx.$selectedItem]);

  const onToggleThumbnails = useCallback(() => {
    dispatch(canvasSessionThumbnailsVisibilityToggled());
  }, [dispatch]);

  return (
    <Flex
      position="absolute"
      flexDir="column"
      bottom={2}
      gap={2}
      align="center"
      justify="center"
      left={2}
      right={2}
      opacity={isStaging ? 1 : 0}
      visibility={isStaging ? 'visible' : 'hidden'}
      pointerEvents={isStaging ? 'auto' : 'none'}
      transitionProperty="opacity"
      transitionDuration="normal"
      aria-hidden={!isStaging}
    >
      <Flex
        w="full"
        h={areThumbnailsVisible ? STAGING_AREA_THUMBNAIL_STRIP_HEIGHT : 0}
        opacity={areThumbnailsVisible ? 1 : 0}
        overflow="hidden"
        pointerEvents={areThumbnailsVisible ? 'auto' : 'none'}
        transitionProperty="height, opacity"
        transitionDuration="normal"
      >
        <StagingAreaItemsList />
      </Flex>
      <StagingAreaToolbar areThumbnailsVisible={areThumbnailsVisible} onToggleThumbnails={onToggleThumbnails} />
    </Flex>
  );
});
StagingArea.displayName = 'StagingArea';

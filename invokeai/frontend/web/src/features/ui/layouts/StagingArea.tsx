import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { STAGING_AREA_THUMBNAIL_STRIP_HEIGHT } from 'features/controlLayers/components/StagingArea/shared';
import { StagingAreaItemsList } from 'features/controlLayers/components/StagingArea/StagingAreaItemsList';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import {
  canvasSessionThumbnailsVisibilityToggled,
  selectCanvasSessionAreThumbnailsVisible,
  useCanvasIsStaging,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useCallback } from 'react';

export const StagingArea = memo(() => {
  const dispatch = useAppDispatch();
  const isStaging = useCanvasIsStaging();
  const areThumbnailsVisible = useAppSelector(selectCanvasSessionAreThumbnailsVisible);

  const onToggleThumbnails = useCallback(() => {
    dispatch(canvasSessionThumbnailsVisibilityToggled());
  }, [dispatch]);

  if (!isStaging) {
    return null;
  }

  return (
    <Flex position="absolute" flexDir="column" bottom={2} gap={2} align="center" justify="center" left={2} right={2}>
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

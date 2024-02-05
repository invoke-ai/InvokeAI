import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { selectProgressSlice } from 'features/progress/store/progressSlice';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

const selectLatestCanvasImageName = createSelector(selectProgressSlice, selectCanvasSlice, (progress, _canvas) => {
  const { latestDenoiseProgress, latestImageOutputEvent } = progress;

  if (!latestImageOutputEvent) {
    return null;
  }
  if (!progress.canvasBatchIds.includes(latestImageOutputEvent.queue_batch_id)) {
    return null;
  }

  if (
    latestDenoiseProgress &&
    latestDenoiseProgress.graph_execution_state_id === latestImageOutputEvent.graph_execution_state_id
  ) {
    return latestImageOutputEvent.image_name;
  }

  if (!latestDenoiseProgress?.progress_image) {
    return latestImageOutputEvent.image_name;
  }

  return null;
});

/**
 * Returns the latest image's DTO. This is not the currently selected image, just the last image received.
 */
export const useLatestCanvasImageDTO = () => {
  const latestImageName = useAppSelector(selectLatestCanvasImageName);
  const { currentData: imageDTO } = useGetImageDTOQuery(latestImageName ?? skipToken);
  return imageDTO;
};

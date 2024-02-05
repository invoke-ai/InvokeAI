import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectProgressSlice } from 'features/progress/store/progressSlice';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

export const selectLatestImageName = createSelector(selectProgressSlice, (progress) => {
  const { latestDenoiseProgress, latestImageOutputEvent, canvasBatchIds } = progress;

  if (!latestImageOutputEvent) {
    return null;
  }

  if (canvasBatchIds.includes(latestImageOutputEvent.queue_batch_id)) {
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
export const useLatestImageDTO = () => {
  const latestImageName = useAppSelector(selectLatestImageName);
  const { currentData: imageDTO } = useGetImageDTOQuery(latestImageName ?? skipToken);
  return imageDTO;
};

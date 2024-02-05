import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useMemo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

/**
 * Returns the latest image's DTO. This is not the currently selected image, just the last image received.
 */
export const useLatestImageDTO = () => {
  const linearDenoiseProgress = useAppSelector((s) => s.progress.linearDenoiseProgress);
  const linearLatestImageData = useAppSelector((s) => s.progress.linearLatestImageData);

  const shouldShowOutputImage = useMemo(() => {
    if (
      linearDenoiseProgress &&
      linearLatestImageData &&
      linearDenoiseProgress.graph_execution_state_id === linearLatestImageData.graph_execution_state_id
    ) {
      return true;
    }

    if (!linearDenoiseProgress?.progress_image && linearLatestImageData) {
      return true;
    }

    return false;
  }, [linearDenoiseProgress, linearLatestImageData]);

  const { currentData: imageDTO } = useGetImageDTOQuery(
    shouldShowOutputImage && linearLatestImageData ? linearLatestImageData.image_name : skipToken
  );

  return imageDTO;
};

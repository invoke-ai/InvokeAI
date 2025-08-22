import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectMetadataFetchDebounce } from 'features/system/store/configSlice';
import { imagesApi, useGetImageMetadataQuery } from 'services/api/endpoints/images';
import { useGetVideoMetadataQuery, videosApi } from 'services/api/endpoints/videos';
import { useDebounce } from 'use-debounce';

export const useDebouncedMetadata = (imageName?: string | null) => {
  const metadataFetchDebounce = useAppSelector(selectMetadataFetchDebounce);

  const [debouncedImageName] = useDebounce(imageName, metadataFetchDebounce);
  const { currentData: cachedData } = imagesApi.endpoints.getImageMetadata.useQueryState(imageName ?? skipToken);
  const { currentData: data, isFetching } = useGetImageMetadataQuery(debouncedImageName ?? skipToken);

  return {
    metadata: cachedData ?? data,
    isLoading: cachedData ? false : isFetching || imageName !== debouncedImageName,
  };
};

export const useDebouncedVideoMetadata = (videoId?: string | null) => {
  const metadataFetchDebounce = useAppSelector(selectMetadataFetchDebounce);

  const [debouncedVideoId] = useDebounce(videoId, metadataFetchDebounce);
  const { currentData: cachedData } = videosApi.endpoints.getVideoMetadata.useQueryState(videoId ?? skipToken);
  const { currentData: data, isFetching } = useGetVideoMetadataQuery(debouncedVideoId ?? skipToken);

  return {
    metadata: cachedData ?? data,
    isLoading: cachedData ? false : isFetching || videoId !== debouncedVideoId,
  };
};

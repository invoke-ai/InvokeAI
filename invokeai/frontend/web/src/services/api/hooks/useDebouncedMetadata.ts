import { skipToken } from '@reduxjs/toolkit/query';
import { imagesApi, useGetImageMetadataQuery } from 'services/api/endpoints/images';
import { useDebounce } from 'use-debounce';

export const useDebouncedMetadata = (imageName?: string | null) => {
  const [debouncedImageName] = useDebounce(imageName, 300);
  const { currentData: cachedData } = imagesApi.endpoints.getImageMetadata.useQueryState(imageName ?? skipToken);
  const { currentData: data, isFetching } = useGetImageMetadataQuery(debouncedImageName ?? skipToken);

  return {
    metadata: cachedData ?? data,
    isLoading: cachedData ? false : isFetching || imageName !== debouncedImageName,
  };
};

import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectMetadataFetchDebounce } from 'features/system/store/configSlice';
import { imagesApi, useGetImageMetadataQuery } from 'services/api/endpoints/images';
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

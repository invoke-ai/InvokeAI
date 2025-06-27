import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectMetadataFetchDebounce } from 'features/system/store/configSlice';
import { useGetImageMetadataQuery } from 'services/api/endpoints/images';
import { useDebounce } from 'use-debounce';

export const useDebouncedMetadata = (imageName?: string | null) => {
  const metadataFetchDebounce = useAppSelector(selectMetadataFetchDebounce);

  const [debouncedImageName] = useDebounce(imageName, metadataFetchDebounce);
  const { data, isFetching } = useGetImageMetadataQuery(debouncedImageName ?? skipToken);

  return { metadata: data, isLoading: isFetching || imageName !== debouncedImageName };
};

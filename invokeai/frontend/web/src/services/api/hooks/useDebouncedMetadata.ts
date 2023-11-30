import { skipToken } from '@reduxjs/toolkit/query';
import { useDebounce } from 'use-debounce';
import { useGetImageMetadataQuery } from 'services/api/endpoints/images';
import { useAppSelector } from 'app/store/storeHooks';

export const useDebouncedMetadata = (imageName?: string | null) => {
  const metadataFetchDebounce = useAppSelector(
    (state) => state.config.metadataFetchDebounce
  );

  const [debouncedImageName] = useDebounce(
    imageName,
    metadataFetchDebounce ?? 0
  );

  const { data: metadata, isLoading } = useGetImageMetadataQuery(
    debouncedImageName ?? skipToken
  );

  return { metadata, isLoading };
};

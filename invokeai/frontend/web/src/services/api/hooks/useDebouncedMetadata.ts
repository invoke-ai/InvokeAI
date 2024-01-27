import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useGetImageMetadataQuery } from 'services/api/endpoints/images';
import { useDebounce } from 'use-debounce';

export const useDebouncedMetadata = (imageName?: string | null) => {
  const metadataFetchDebounce = useAppSelector((s) => s.config.metadataFetchDebounce ?? 300);

  const [debouncedImageName] = useDebounce(imageName, metadataFetchDebounce);

  const { data: metadata, isLoading } = useGetImageMetadataQuery(debouncedImageName ?? skipToken);

  return { metadata, isLoading };
};

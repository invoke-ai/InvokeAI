import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectGalleryView,
  selectGetImageNamesQueryArgs,
  selectGetVideoIdsQueryArgs,
} from 'features/gallery/store/gallerySelectors';
import { useGetImageNamesQuery } from 'services/api/endpoints/images';
import { useGetVideoIdsQuery } from 'services/api/endpoints/videos';
import { useDebounce } from 'use-debounce';

const getImageNamesQueryOptions = {
  refetchOnReconnect: true,
  selectFromResult: ({ currentData, isLoading, isFetching }) => ({
    imageNames: currentData?.image_names ?? EMPTY_ARRAY,
    isLoading,
    isFetching,
  }),
} satisfies Parameters<typeof useGetImageNamesQuery>[1];

const getVideoIdsQueryOptions = {
  refetchOnReconnect: true,
  selectFromResult: ({ currentData, isLoading, isFetching }) => ({
    video_ids: currentData?.video_ids ?? EMPTY_ARRAY,
    isLoading,
    isFetching,
  }),
} satisfies Parameters<typeof useGetVideoIdsQuery>[1];

export const useGalleryImageNames = () => {
  const galleryView = useAppSelector(selectGalleryView);
  const _queryArgs = useAppSelector(selectGetImageNamesQueryArgs);
  const [queryArgs] = useDebounce(_queryArgs, 300);
  const { imageNames, isLoading, isFetching } = useGetImageNamesQuery(
    galleryView !== 'videos' ? queryArgs : skipToken,
    getImageNamesQueryOptions
  );
  return { imageNames, isLoading, isFetching, queryArgs };
};

export const useGalleryVideoIds = () => {
  const galleryView = useAppSelector(selectGalleryView);
  const _queryArgs = useAppSelector(selectGetVideoIdsQueryArgs);
  const [queryArgs] = useDebounce(_queryArgs, 300);
  const { video_ids, isLoading, isFetching } = useGetVideoIdsQuery(
    galleryView === 'videos' ? queryArgs : skipToken,
    getVideoIdsQueryOptions
  );
  return { video_ids, isLoading, isFetching, queryArgs };
};

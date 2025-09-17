import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectGetVideoIdsQueryArgs } from 'features/gallery/store/gallerySelectors';
import { selectAllowVideo } from 'features/system/store/configSlice';
import { useGetVideoIdsQuery } from 'services/api/endpoints/videos';
import { useDebounce } from 'use-debounce';

const getVideoIdsQueryOptions = {
  refetchOnReconnect: true,
  selectFromResult: ({ currentData, isLoading, isFetching }) => ({
    videoIds: currentData?.video_ids ?? EMPTY_ARRAY,
    isLoading,
    isFetching,
  }),
} satisfies Parameters<typeof useGetVideoIdsQuery>[1];

export const useGalleryVideoIds = () => {
  const isVideoEnabled = useAppSelector(selectAllowVideo);
  const _queryArgs = useAppSelector(selectGetVideoIdsQueryArgs);
  const [queryArgs] = useDebounce(_queryArgs, 300);
  const { videoIds, isLoading, isFetching } = useGetVideoIdsQuery(queryArgs, {
    ...getVideoIdsQueryOptions,
    skip: !isVideoEnabled,
  });
  return { videoIds, isLoading, isFetching, queryArgs };
};

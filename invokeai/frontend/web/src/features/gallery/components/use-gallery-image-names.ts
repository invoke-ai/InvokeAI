import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectGetImageNamesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { useGetImageNamesQuery } from 'services/api/endpoints/images';
import { useDebounce } from 'use-debounce';

const getImageNamesQueryOptions = {
  refetchOnReconnect: true,
  selectFromResult: ({ currentData, isLoading, isFetching }) => ({
    imageNames: currentData?.image_names ?? EMPTY_ARRAY,
    isLoading,
    isFetching,
  }),
} satisfies Parameters<typeof useGetImageNamesQuery>[1];

export const useGalleryImageNames = () => {
  const _queryArgs = useAppSelector(selectGetImageNamesQueryArgs);
  const [queryArgs] = useDebounce(_queryArgs, 300);
  const { imageNames, isLoading, isFetching } = useGetImageNamesQuery(queryArgs, getImageNamesQueryOptions);
  return { imageNames, isLoading, isFetching, queryArgs };
};

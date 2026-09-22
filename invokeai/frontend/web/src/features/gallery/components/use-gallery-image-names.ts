import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectGalleryItemNamesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { useListGalleryItemNamesQuery } from 'services/api/endpoints/gallery';
import { useDebounce } from 'use-debounce';

const selectFromGalleryItemNamesResult = ({
  currentData,
  isLoading,
  isFetching,
}: {
  currentData?: { item_names: string[] };
  isLoading: boolean;
  isFetching: boolean;
}) => ({
  imageNames: currentData?.item_names ?? (EMPTY_ARRAY as string[]),
  isLoading,
  isFetching,
});

const galleryQueryOptions = {
  refetchOnReconnect: true,
  selectFromResult: selectFromGalleryItemNamesResult,
};

/**
 * Returns the ordered flat list of gallery item names. Names are polymorphic — both image and
 * video names appear in the same list, interleaved by created_at. Callers that need to know the
 * kind of a particular name use `isVideoName` from `features/gallery/store/types`.
 *
 * Regular boards and date-based virtual boards share one endpoint; the selector translates a
 * virtual board id into the `created_date` filter.
 */
export const useGalleryImageNames = () => {
  const _queryArgs = useAppSelector(selectGalleryItemNamesQueryArgs);
  const [queryArgs] = useDebounce(_queryArgs, 300);

  const { imageNames, isLoading, isFetching } = useListGalleryItemNamesQuery(queryArgs, galleryQueryOptions);

  return { imageNames, isLoading, isFetching, queryArgs };
};

import { skipToken } from '@reduxjs/toolkit/query';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectGetImageNamesQueryArgs, selectSelectedBoardId } from 'features/gallery/store/gallerySelectors';
import { getDateFromVirtualBoardId, isVirtualBoardId } from 'features/gallery/store/types';
import { useGetImageNamesQuery } from 'services/api/endpoints/images';
import { useGetVirtualBoardImageNamesByDateQuery } from 'services/api/endpoints/virtual_boards';
import { useDebounce } from 'use-debounce';

const selectFromResult = ({
  currentData,
  isLoading,
  isFetching,
}: {
  currentData?: { image_names: string[] };
  isLoading: boolean;
  isFetching: boolean;
}) => ({
  imageNames: currentData?.image_names ?? EMPTY_ARRAY,
  isLoading,
  isFetching,
});

const queryOptions = {
  refetchOnReconnect: true,
  selectFromResult,
};

export const useGalleryImageNames = () => {
  const selectedBoardId = useAppSelector(selectSelectedBoardId);
  const _queryArgs = useAppSelector(selectGetImageNamesQueryArgs);
  const [queryArgs] = useDebounce(_queryArgs, 300);
  const isVirtual = isVirtualBoardId(selectedBoardId);

  // Regular board query
  const regularResult = useGetImageNamesQuery(isVirtual ? skipToken : queryArgs, queryOptions);

  // Virtual board query
  const date = isVirtual ? getDateFromVirtualBoardId(selectedBoardId) : '';
  const virtualResult = useGetVirtualBoardImageNamesByDateQuery(
    isVirtual
      ? {
          date,
          categories: queryArgs.categories ?? undefined,
          search_term: queryArgs.search_term || undefined,
          order_dir: queryArgs.order_dir,
          starred_first: queryArgs.starred_first,
        }
      : skipToken,
    queryOptions
  );

  const result = isVirtual ? virtualResult : regularResult;

  return {
    imageNames: result.imageNames,
    isLoading: result.isLoading,
    isFetching: result.isFetching,
    queryArgs,
  };
};

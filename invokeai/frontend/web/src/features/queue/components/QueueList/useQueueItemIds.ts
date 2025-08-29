import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectGetQueueItemIdsArgs } from 'features/queue/store/queueSlice';
import { useGetQueueItemIdsQuery } from 'services/api/endpoints/queue';
import { useDebounce } from 'use-debounce';

const getQueueItemIdsQueryOptions = {
  refetchOnReconnect: true,
  selectFromResult: ({ currentData, isLoading, isFetching }) => ({
    item_ids: currentData?.item_ids ?? EMPTY_ARRAY,
    isLoading,
    isFetching,
  }),
} satisfies Parameters<typeof useGetQueueItemIdsQuery>[1];

export const useQueueItemIds = () => {
  const _queryArgs = useAppSelector(selectGetQueueItemIdsArgs);
  const [queryArgs] = useDebounce(_queryArgs, 300);
  const { item_ids, isLoading, isFetching } = useGetQueueItemIdsQuery(queryArgs, getQueueItemIdsQueryOptions);
  return { queryArgs, itemIds: item_ids, isLoading, isFetching };
};

import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { BsStars } from 'react-icons/bs';
import {
  useGetQueueStatusQuery,
  usePruneQueueMutation,
} from 'services/api/endpoints/queue';
import { useIsQueueMutationInProgress } from '../hooks/useIsQueueMutationInProgress';
import { listCursorChanged, listPriorityChanged } from '../store/queueSlice';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const PruneQueueButton = ({ asIconButton }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [pruneQueue] = usePruneQueueMutation({ fixedCacheKey: 'pruneQueue' });
  const isQueueMutationInProgress = useIsQueueMutationInProgress();
  const { count } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { count: 0 };
      }

      return { count: data.completed + data.canceled + data.failed };
    },
  });

  const handleClick = useCallback(async () => {
    try {
      const data = await pruneQueue().unwrap();
      dispatch(
        addToast({
          title: t('queue.pruneSucceeded', { item_count: data.deleted }),
          status: 'success',
        })
      );
      dispatch(listCursorChanged(undefined));
      dispatch(listPriorityChanged(undefined));
    } catch {
      dispatch(
        addToast({
          title: t('queue.pruneFailed'),
          status: 'error',
        })
      );
    }
  }, [dispatch, pruneQueue, t]);

  return (
    <QueueButton
      isDisabled={!count || isQueueMutationInProgress}
      asIconButton={asIconButton}
      label={t('queue.prune')}
      tooltip={t('queue.pruneTooltip', { item_count: count })}
      icon={<BsStars />}
      onClick={handleClick}
      colorScheme="blue"
    />
  );
};

export default memo(PruneQueueButton);

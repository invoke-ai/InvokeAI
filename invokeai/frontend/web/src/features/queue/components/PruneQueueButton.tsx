import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaScissors } from 'react-icons/fa6';
import {
  useGetQueueStatusQuery,
  usePruneQueueMutation,
} from 'services/api/endpoints/queue';
import { listCursorChanged, listPriorityChanged } from '../store/queueSlice';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const PruneQueueButton = ({ asIconButton }: Props) => {
  const dispatch = useAppDispatch();
  const { count } = useGetQueueStatusQuery(undefined, {
    selectFromResult: ({ data }) => {
      if (!data) {
        return { count: 0 };
      }

      return { count: data.completed + data.canceled + data.failed };
    },
  });
  const { t } = useTranslation();
  const [pruneQueue] = usePruneQueueMutation();
  const handleClick = useCallback(async () => {
    try {
      const data = await pruneQueue().unwrap();
      dispatch(
        addToast({
          title: t('queue.pruneSucceeded', { count: data.deleted }),
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
      isDisabled={!count}
      asIconButton={asIconButton}
      label={t('queue.prune')}
      tooltip={t('queue.pruneTooltip', { count })}
      icon={<FaScissors />}
      onClick={handleClick}
      colorScheme="blue"
    />
  );
};

export default memo(PruneQueueButton);

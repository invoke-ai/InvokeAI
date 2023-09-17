import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import {
  useClearQueueMutation,
  useGetQueueStatusQuery,
} from 'services/api/endpoints/queue';
import { listCursorChanged, listPriorityChanged } from '../store/queueSlice';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const ClearQueueButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();

  const dispatch = useAppDispatch();
  const { data: queueStatus } = useGetQueueStatusQuery();

  const [clearQueue, { isLoading }] = useClearQueueMutation({
    fixedCacheKey: 'clearQueue',
  });

  const handleClick = useCallback(async () => {
    try {
      await clearQueue().unwrap();
      dispatch(
        addToast({
          title: t('queue.clearSucceeded'),
          status: 'success',
        })
      );
      dispatch(listCursorChanged(undefined));
      dispatch(listPriorityChanged(undefined));
    } catch {
      dispatch(
        addToast({
          title: t('queue.clearFailed'),
          status: 'error',
        })
      );
    }
  }, [clearQueue, dispatch, t]);

  return (
    <QueueButton
      isDisabled={!queueStatus?.queue.total}
      isLoading={isLoading}
      asIconButton={asIconButton}
      label={t('queue.clear')}
      tooltip={t('queue.clearTooltip')}
      icon={<FaTrash />}
      onClick={handleClick}
      colorScheme="error"
    />
  );
};

export default memo(ClearQueueButton);

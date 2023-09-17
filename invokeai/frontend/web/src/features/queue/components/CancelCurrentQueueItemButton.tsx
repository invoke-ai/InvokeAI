import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTimes } from 'react-icons/fa';
import {
  useCancelQueueItemMutation,
  useGetQueueStatusQuery,
} from 'services/api/endpoints/queue';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const CancelCurrentQueueItemButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const [cancelQueueItem, { isLoading }] = useCancelQueueItemMutation({
    fixedCacheKey: 'cancelQueueItem',
  });

  const handleClick = useCallback(async () => {
    if (!queueStatus?.queue.item_id) {
      return;
    }
    try {
      await cancelQueueItem(queueStatus.queue.item_id).unwrap();
      dispatch(
        addToast({
          title: t('queue.cancelSucceeded'),
          status: 'success',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('queue.cancelFailed'),
          status: 'error',
        })
      );
    }
  }, [cancelQueueItem, dispatch, queueStatus?.queue.item_id, t]);

  return (
    <QueueButton
      isDisabled={!queueStatus?.queue.item_id}
      isLoading={isLoading}
      asIconButton={asIconButton}
      label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<FaTimes />}
      onClick={handleClick}
      colorScheme="error"
    />
  );
};

export default memo(CancelCurrentQueueItemButton);

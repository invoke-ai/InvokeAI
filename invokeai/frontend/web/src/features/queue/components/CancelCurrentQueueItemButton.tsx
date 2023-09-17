import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTimes } from 'react-icons/fa';
import {
  useCancelQueueItemMutation,
  useGetCurrentQueueItemQuery,
} from 'services/api/endpoints/queue';
import { useIsQueueMutationInProgress } from '../hooks/useIsQueueMutationInProgress';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const CancelCurrentQueueItemButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { data: currentQueueItem } = useGetCurrentQueueItemQuery();
  const [cancelQueueItem] = useCancelQueueItemMutation({
    fixedCacheKey: 'cancelQueueItem',
  });
  const isQueueMutationInProgress = useIsQueueMutationInProgress();

  const handleClick = useCallback(async () => {
    if (!currentQueueItem) {
      return;
    }
    try {
      await cancelQueueItem(currentQueueItem.item_id).unwrap();
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
  }, [cancelQueueItem, currentQueueItem, dispatch, t]);

  return (
    <QueueButton
      asIconButton={asIconButton}
      label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      isDisabled={!currentQueueItem || isQueueMutationInProgress}
      icon={<FaTimes />}
      onClick={handleClick}
      colorScheme="error"
    />
  );
};

export default memo(CancelCurrentQueueItemButton);

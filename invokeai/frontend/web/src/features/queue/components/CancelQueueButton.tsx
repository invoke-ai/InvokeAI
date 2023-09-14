import { useAppDispatch } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTimes } from 'react-icons/fa';
import {
  useCancelQueueExecutionMutation,
  useGetQueueStatusQuery,
} from 'services/api/endpoints/queue';
import QueueButton from './common/QueueButton';
import { addToast } from 'features/system/store/systemSlice';
import { useIsQueueMutationInProgress } from '../hooks/useIsQueueMutationInProgress';

type Props = {
  asIconButton?: boolean;
};

const CancelQueueButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { data: queueStatusData } = useGetQueueStatusQuery();
  const [cancelQueue] = useCancelQueueExecutionMutation({
    fixedCacheKey: 'cancelQueue',
  });
  const isQueueMutationInProgress = useIsQueueMutationInProgress();

  const handleClick = useCallback(async () => {
    try {
      await cancelQueue().unwrap();
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
  }, [cancelQueue, dispatch, t]);

  return (
    <QueueButton
      asIconButton={asIconButton}
      label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      isDisabled={
        !(queueStatusData?.started || queueStatusData?.stop_after_current) ||
        isQueueMutationInProgress
      }
      icon={<FaTimes />}
      onClick={handleClick}
      colorScheme="orange"
    />
  );
};

export default memo(CancelQueueButton);

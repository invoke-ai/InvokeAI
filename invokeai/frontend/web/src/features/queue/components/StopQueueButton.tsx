import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaStop } from 'react-icons/fa';
import {
  useGetQueueStatusQuery,
  useStopQueueExecutionMutation,
} from 'services/api/endpoints/queue';
import QueueButton from './common/QueueButton';
import { addToast } from 'features/system/store/systemSlice';
import { useAppDispatch } from 'app/store/storeHooks';

type Props = {
  asIconButton?: boolean;
};

const StopQueueButton = ({ asIconButton }: Props) => {
  const { data: queueStatusData } = useGetQueueStatusQuery();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [stopQueue] = useStopQueueExecutionMutation();
  const handleClick = useCallback(async () => {
    try {
      await stopQueue().unwrap();
      dispatch(
        addToast({
          title: t('queue.stopRequested'),
          status: 'info',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('queue.stopFailed'),
          status: 'error',
        })
      );
    }
  }, [dispatch, stopQueue, t]);

  return (
    <QueueButton
      asIconButton={asIconButton}
      label={t('queue.stop')}
      tooltip={t('queue.stopTooltip')}
      isDisabled={!queueStatusData?.started}
      icon={<FaStop />}
      onClick={handleClick}
      colorScheme="gold"
    />
  );
};

export default memo(StopQueueButton);

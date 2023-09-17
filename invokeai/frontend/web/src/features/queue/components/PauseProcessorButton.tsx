import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaPause } from 'react-icons/fa';
import {
  useGetQueueStatusQuery,
  usePauseProcessorMutation,
} from 'services/api/endpoints/queue';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const PauseProcessorButton = ({ asIconButton }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const [pauseProcessor, { isLoading }] = usePauseProcessorMutation({
    fixedCacheKey: 'pauseProcessor',
  });

  const handleClick = useCallback(async () => {
    try {
      await pauseProcessor().unwrap();
      dispatch(
        addToast({
          title: t('queue.pauseRequested'),
          status: 'info',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('queue.pauseFailed'),
          status: 'error',
        })
      );
    }
  }, [dispatch, pauseProcessor, t]);

  return (
    <QueueButton
      asIconButton={asIconButton}
      label={t('queue.pause')}
      tooltip={t('queue.pauseTooltip')}
      isDisabled={!queueStatus?.processor.is_started}
      isLoading={isLoading}
      icon={<FaPause />}
      onClick={handleClick}
      colorScheme="gold"
    />
  );
};

export default memo(PauseProcessorButton);

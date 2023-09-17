import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaStop } from 'react-icons/fa';
import {
  useGetProcessorStatusQuery,
  usePauseProcessorMutation,
} from 'services/api/endpoints/queue';
import { useIsQueueMutationInProgress } from '../hooks/useIsQueueMutationInProgress';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const PauseProcessorButton = ({ asIconButton }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { data: processorStatus } = useGetProcessorStatusQuery();
  const [pauseProcessor] = usePauseProcessorMutation({
    fixedCacheKey: 'pauseProcessor',
  });
  const isQueueMutationInProgress = useIsQueueMutationInProgress();

  const handleClick = useCallback(async () => {
    try {
      await pauseProcessor().unwrap();
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
  }, [dispatch, pauseProcessor, t]);

  return (
    <QueueButton
      asIconButton={asIconButton}
      label={t('queue.stop')}
      tooltip={t('queue.stopTooltip')}
      isDisabled={!processorStatus?.is_started || isQueueMutationInProgress}
      isLoading={processorStatus?.is_stop_pending}
      icon={<FaStop />}
      onClick={handleClick}
      colorScheme="gold"
    />
  );
};

export default memo(PauseProcessorButton);

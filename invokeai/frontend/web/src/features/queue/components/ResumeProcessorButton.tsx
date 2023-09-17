import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaPlay } from 'react-icons/fa';
import {
  useGetProcessorStatusQuery,
  useResumeProcessorMutation,
} from 'services/api/endpoints/queue';
import { useIsQueueMutationInProgress } from '../hooks/useIsQueueMutationInProgress';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const ResumeProcessorButton = ({ asIconButton }: Props) => {
  const { data: processorStatus } = useGetProcessorStatusQuery();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [resumeProcessor] = useResumeProcessorMutation({
    fixedCacheKey: 'resumeProcessor',
  });
  const isQueueMutationInProgress = useIsQueueMutationInProgress();

  const handleClick = useCallback(async () => {
    try {
      await resumeProcessor().unwrap();
      dispatch(
        addToast({
          title: t('queue.resumeSucceeded'),
          status: 'success',
        })
      );
    } catch {
      dispatch(
        addToast({
          title: t('queue.resumeFailed'),
          status: 'error',
        })
      );
    }
  }, [dispatch, resumeProcessor, t]);

  return (
    <QueueButton
      asIconButton={asIconButton}
      label={t('queue.resume')}
      tooltip={t('queue.resumeTooltip')}
      isDisabled={
        processorStatus?.is_started ||
        processorStatus?.is_processing ||
        isQueueMutationInProgress
      }
      icon={<FaPlay />}
      onClick={handleClick}
      colorScheme="green"
    />
  );
};

export default memo(ResumeProcessorButton);

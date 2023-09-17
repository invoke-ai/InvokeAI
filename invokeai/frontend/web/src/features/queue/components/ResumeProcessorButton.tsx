import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaPlay } from 'react-icons/fa';
import {
  useGetQueueStatusQuery,
  useResumeProcessorMutation,
} from 'services/api/endpoints/queue';
import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const ResumeProcessorButton = ({ asIconButton }: Props) => {
  const { data: queueStatus } = useGetQueueStatusQuery();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [resumeProcessor, { isLoading }] = useResumeProcessorMutation({
    fixedCacheKey: 'resumeProcessor',
  });

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
      isDisabled={queueStatus?.processor.is_started}
      isLoading={isLoading}
      icon={<FaPlay />}
      onClick={handleClick}
      colorScheme="green"
    />
  );
};

export default memo(ResumeProcessorButton);

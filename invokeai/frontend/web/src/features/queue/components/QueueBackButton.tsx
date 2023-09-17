import IAIButton from 'common/components/IAIButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useQueueBack } from '../hooks/useQueueBack';
import EnqueueButtonTooltip from './QueueButtonTooltip';

const QueueBackButton = () => {
  const { t } = useTranslation();
  const { queueBack, isLoading, isDisabled } = useQueueBack();
  return (
    <IAIButton
      isDisabled={isDisabled}
      isLoading={isLoading}
      colorScheme="accent"
      onClick={queueBack}
      tooltip={<EnqueueButtonTooltip />}
      flexGrow={3}
      minW={44}
    >
      {t('queue.queueBack')}
    </IAIButton>
  );
};

export default memo(QueueBackButton);

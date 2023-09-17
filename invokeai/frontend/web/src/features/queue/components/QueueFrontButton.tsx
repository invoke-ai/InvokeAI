import IAIIconButton from 'common/components/IAIIconButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaBoltLightning } from 'react-icons/fa6';
import { useQueueFront } from '../hooks/useQueueFront';
import EnqueueButtonTooltip from './QueueButtonTooltip';

const QueueFrontButton = () => {
  const { t } = useTranslation();
  const { queueFront, isLoading, isDisabled } = useQueueFront();
  return (
    <IAIIconButton
      colorScheme="base"
      aria-label={t('queue.queueFront')}
      isDisabled={isDisabled}
      isLoading={isLoading}
      onClick={queueFront}
      tooltip={<EnqueueButtonTooltip prepend />}
      icon={<FaBoltLightning />}
    />
  );
};

export default memo(QueueFrontButton);

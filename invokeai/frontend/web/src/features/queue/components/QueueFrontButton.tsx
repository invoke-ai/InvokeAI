import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { useQueueFront } from 'features/queue/hooks/useQueueFront';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaBoltLightning } from 'react-icons/fa6';

import { QueueButtonTooltip } from './QueueButtonTooltip';

const QueueFrontButton = () => {
  const { t } = useTranslation();
  const { queueFront, isLoading, isDisabled } = useQueueFront();
  return (
    <InvIconButton
      aria-label={t('queue.queueFront')}
      isDisabled={isDisabled}
      isLoading={isLoading}
      onClick={queueFront}
      tooltip={<QueueButtonTooltip prepend />}
      icon={<FaBoltLightning />}
      size="lg"
    />
  );
};

export default memo(QueueFrontButton);

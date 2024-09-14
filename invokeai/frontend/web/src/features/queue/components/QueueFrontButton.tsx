import { IconButton } from '@invoke-ai/ui-library';
import { useQueueFront } from 'features/queue/hooks/useQueueFront';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLightningFill } from 'react-icons/pi';

import { QueueButtonTooltip } from './QueueButtonTooltip';

const QueueFrontButton = () => {
  const { t } = useTranslation();
  const { queueFront, isLoading, isDisabled } = useQueueFront();
  return (
    <QueueButtonTooltip prepend>
      <IconButton
        aria-label={t('queue.queueFront')}
        isDisabled={isDisabled}
        isLoading={isLoading}
        onClick={queueFront}
        icon={<PiLightningFill />}
        size="lg"
      />
    </QueueButtonTooltip>
  );
};

export default memo(QueueFrontButton);

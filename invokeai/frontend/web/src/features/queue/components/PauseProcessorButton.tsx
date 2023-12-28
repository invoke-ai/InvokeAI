import { usePauseProcessor } from 'features/queue/hooks/usePauseProcessor';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaPause } from 'react-icons/fa';

import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const PauseProcessorButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const { pauseProcessor, isLoading, isDisabled } = usePauseProcessor();

  return (
    <QueueButton
      asIconButton={asIconButton}
      label={t('queue.pause')}
      tooltip={t('queue.pauseTooltip')}
      isDisabled={isDisabled}
      isLoading={isLoading}
      icon={<FaPause />}
      onClick={pauseProcessor}
      colorScheme="gold"
    />
  );
};

export default memo(PauseProcessorButton);

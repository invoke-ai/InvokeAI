import { usePauseProcessor } from 'features/queue/hooks/usePauseProcessor';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPauseFill } from 'react-icons/pi';

import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const PauseProcessorButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const pauseProcessor = usePauseProcessor();

  return (
    <QueueButton
      asIconButton={asIconButton}
      label={t('queue.pause')}
      tooltip={t('queue.pauseTooltip')}
      isDisabled={pauseProcessor.isDisabled}
      isLoading={pauseProcessor.isLoading}
      icon={<PiPauseFill />}
      onClick={pauseProcessor.trigger}
      colorScheme="gold"
    />
  );
};

export default memo(PauseProcessorButton);

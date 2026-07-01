import { useResumeProcessor } from 'features/queue/hooks/useResumeProcessor';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlayFill } from 'react-icons/pi';

import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
};

const ResumeProcessorButton = ({ asIconButton }: Props) => {
  const { t } = useTranslation();
  const resumeProcessor = useResumeProcessor();

  return (
    <QueueButton
      asIconButton={asIconButton}
      label={t('queue.resume')}
      tooltip={t('queue.resumeTooltip')}
      isDisabled={resumeProcessor.isDisabled}
      isLoading={resumeProcessor.isLoading}
      icon={<PiPlayFill />}
      onClick={resumeProcessor.trigger}
      colorScheme="green"
    />
  );
};

export default memo(ResumeProcessorButton);

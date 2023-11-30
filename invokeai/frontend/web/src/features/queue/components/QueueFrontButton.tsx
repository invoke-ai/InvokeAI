import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaBoltLightning } from 'react-icons/fa6';
import { useQueueFront } from 'features/queue/hooks/useQueueFront';
import EnqueueButtonTooltip from './QueueButtonTooltip';
import QueueButton from './common/QueueButton';
import { ChakraProps } from '@chakra-ui/react';

type Props = {
  asIconButton?: boolean;
  sx?: ChakraProps['sx'];
};

const QueueFrontButton = ({ asIconButton, sx }: Props) => {
  const { t } = useTranslation();
  const { queueFront, isLoading, isDisabled } = useQueueFront();
  return (
    <QueueButton
      asIconButton={asIconButton}
      colorScheme="base"
      label={t('queue.queueFront')}
      isDisabled={isDisabled}
      isLoading={isLoading}
      onClick={queueFront}
      tooltip={<EnqueueButtonTooltip prepend />}
      icon={<FaBoltLightning />}
      sx={sx}
    />
  );
};

export default memo(QueueFrontButton);

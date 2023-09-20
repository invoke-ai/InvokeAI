import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import { useClearQueue } from '../hooks/useClearQueue';
import QueueButton from './common/QueueButton';
import { ChakraProps } from '@chakra-ui/react';

type Props = {
  asIconButton?: boolean;
  sx?: ChakraProps['sx'];
};

const ClearQueueButton = ({ asIconButton, sx }: Props) => {
  const { t } = useTranslation();
  const { clearQueue, isLoading, queueStatus } = useClearQueue();

  return (
    <QueueButton
      isDisabled={!queueStatus?.queue.total}
      isLoading={isLoading}
      asIconButton={asIconButton}
      label={t('queue.clear')}
      tooltip={t('queue.clearTooltip')}
      icon={<FaTrash />}
      onClick={clearQueue}
      colorScheme="error"
      sx={sx}
    />
  );
};

export default memo(ClearQueueButton);

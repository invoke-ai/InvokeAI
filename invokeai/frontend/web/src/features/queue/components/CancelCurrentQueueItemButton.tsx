import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTimes } from 'react-icons/fa';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import QueueButton from './common/QueueButton';
import { ChakraProps } from '@chakra-ui/react';

type Props = {
  asIconButton?: boolean;
  sx?: ChakraProps['sx'];
};

const CancelCurrentQueueItemButton = ({ asIconButton, sx }: Props) => {
  const { t } = useTranslation();
  const { cancelQueueItem, isLoading, isDisabled } =
    useCancelCurrentQueueItem();

  return (
    <QueueButton
      isDisabled={isDisabled}
      isLoading={isLoading}
      asIconButton={asIconButton}
      label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<FaTimes />}
      onClick={cancelQueueItem}
      colorScheme="error"
      sx={sx}
    />
  );
};

export default memo(CancelCurrentQueueItemButton);

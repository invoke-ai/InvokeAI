import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import { useClearQueue } from '../hooks/useClearQueue';
import QueueButton from './common/QueueButton';
import { ChakraProps, Text } from '@chakra-ui/react';
import IAIAlertDialog from 'common/components/IAIAlertDialog';

type Props = {
  asIconButton?: boolean;
  sx?: ChakraProps['sx'];
};

const ClearQueueButton = ({ asIconButton, sx }: Props) => {
  const { t } = useTranslation();
  const { clearQueue, isLoading, isDisabled } = useClearQueue();

  return (
    <IAIAlertDialog
      title={t('queue.clearTooltip')}
      acceptCallback={clearQueue}
      acceptButtonText={t('queue.clear')}
      triggerComponent={
        <QueueButton
          isDisabled={isDisabled}
          isLoading={isLoading}
          asIconButton={asIconButton}
          label={t('queue.clear')}
          tooltip={t('queue.clearTooltip')}
          icon={<FaTrash />}
          colorScheme="error"
          sx={sx}
        />
      }
    >
      <Text>{t('queue.clearQueueAlertDialog')}</Text>
      <br />
      <Text>{t('queue.clearQueueAlertDialog2')}</Text>
    </IAIAlertDialog>
  );
};

export default memo(ClearQueueButton);

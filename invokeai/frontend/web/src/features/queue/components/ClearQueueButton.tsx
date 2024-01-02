import { type ChakraProps, useDisclosure } from '@chakra-ui/react';
import { InvConfirmationAlertDialog } from 'common/components/InvConfirmationAlertDialog/InvConfirmationAlertDialog';
import { InvText } from 'common/components/InvText/wrapper';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';

import QueueButton from './common/QueueButton';

type Props = {
  asIconButton?: boolean;
  sx?: ChakraProps['sx'];
};

const ClearQueueButton = ({ asIconButton, sx }: Props) => {
  const { t } = useTranslation();
  const { isOpen, onClose, onOpen } = useDisclosure();
  const { clearQueue, isLoading, isDisabled } = useClearQueue();

  return (
    <>
      <QueueButton
        isDisabled={isDisabled}
        isLoading={isLoading}
        asIconButton={asIconButton}
        label={t('queue.clear')}
        tooltip={t('queue.clearTooltip')}
        icon={<FaTrash />}
        colorScheme="error"
        sx={sx}
        onClick={onOpen}
      />
      <InvConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('queue.clearTooltip')}
        acceptCallback={clearQueue}
        acceptButtonText={t('queue.clear')}
      >
        <InvText>{t('queue.clearQueueAlertDialog')}</InvText>
        <br />
        <InvText>{t('queue.clearQueueAlertDialog2')}</InvText>
      </InvConfirmationAlertDialog>
    </>
  );
};

export default memo(ClearQueueButton);

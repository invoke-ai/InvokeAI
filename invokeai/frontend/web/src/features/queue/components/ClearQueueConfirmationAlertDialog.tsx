import type {
  UseDisclosureReturn} from '@chakra-ui/react';
import { InvConfirmationAlertDialog } from 'common/components/InvConfirmationAlertDialog/InvConfirmationAlertDialog';
import { InvText } from 'common/components/InvText/wrapper';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  disclosure: UseDisclosureReturn;
};

const ClearQueueButton = ({ disclosure }: Props) => {
  const { t } = useTranslation();
  const { clearQueue } = useClearQueue();

  return (
    <InvConfirmationAlertDialog
      isOpen={disclosure.isOpen}
      onClose={disclosure.onClose}
      title={t('queue.clearTooltip')}
      acceptCallback={clearQueue}
      acceptButtonText={t('queue.clear')}
    >
      <InvText>{t('queue.clearQueueAlertDialog')}</InvText>
      <br />
      <InvText>{t('queue.clearQueueAlertDialog2')}</InvText>
    </InvConfirmationAlertDialog>
  );
};

export default memo(ClearQueueButton);

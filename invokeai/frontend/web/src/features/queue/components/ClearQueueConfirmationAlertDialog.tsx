import type { UseDisclosureReturn } from '@invoke-ai/ui-library';
import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
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
    <ConfirmationAlertDialog
      isOpen={disclosure.isOpen}
      onClose={disclosure.onClose}
      title={t('queue.clearTooltip')}
      acceptCallback={clearQueue}
      acceptButtonText={t('queue.clear')}
    >
      <Text>{t('queue.clearQueueAlertDialog')}</Text>
      <br />
      <Text>{t('queue.clearQueueAlertDialog2')}</Text>
    </ConfirmationAlertDialog>
  );
};

export default memo(ClearQueueButton);

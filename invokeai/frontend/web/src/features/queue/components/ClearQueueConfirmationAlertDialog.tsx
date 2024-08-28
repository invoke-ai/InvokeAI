import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { atom } from 'nanostores';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const $boolean = atom(false);
export const useClearQueueConfirmationAlertDialog = buildUseBoolean($boolean);

export const ClearQueueConfirmationsAlertDialog = memo(() => {
  const { t } = useTranslation();
  const dialogState = useClearQueueConfirmationAlertDialog();
  const isOpen = useStore(dialogState.$boolean);
  const { clearQueue } = useClearQueue();

  return (
    <ConfirmationAlertDialog
      isOpen={isOpen}
      onClose={dialogState.setFalse}
      title={t('queue.clearTooltip')}
      acceptCallback={clearQueue}
      acceptButtonText={t('queue.clear')}
      useInert={false}
    >
      <Text>{t('queue.clearQueueAlertDialog')}</Text>
      <br />
      <Text>{t('queue.clearQueueAlertDialog2')}</Text>
    </ConfirmationAlertDialog>
  );
});

ClearQueueConfirmationsAlertDialog.displayName = 'ClearQueueConfirmationsAlertDialog';

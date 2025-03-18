import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button } from '@invoke-ai/ui-library';
import { useCancelAllExceptCurrentQueueItemDialog } from 'features/queue/components/CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXCircle } from 'react-icons/pi';

type Props = ButtonProps;

export const CancelAllExceptCurrentButton = memo((props: Props) => {
  const { t } = useTranslation();
  const cancelAllExceptCurrent = useCancelAllExceptCurrentQueueItemDialog();

  return (
    <>
      <Button
        onClick={cancelAllExceptCurrent.openDialog}
        isLoading={cancelAllExceptCurrent.isLoading}
        isDisabled={cancelAllExceptCurrent.isDisabled}
        tooltip={t('queue.cancelAllExceptCurrentTooltip')}
        leftIcon={<PiXCircle />}
        colorScheme="error"
        data-testid={t('queue.clear')}
        {...props}
      >
        {t('queue.clear')}
      </Button>
    </>
  );
});

CancelAllExceptCurrentButton.displayName = 'CancelAllExceptCurrentButton';

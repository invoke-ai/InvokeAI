import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashBold } from 'react-icons/pi';

import { useClearQueueDialog } from './ClearQueueConfirmationAlertDialog';

export const ClearQueueButton = memo((props: ButtonProps) => {
  const { t } = useTranslation();
  const api = useClearQueueDialog();

  return (
    <Button
      isDisabled={api.isDisabled}
      isLoading={api.isLoading}
      aria-label={t('queue.clear')}
      tooltip={t('queue.clearTooltip')}
      leftIcon={<PiTrashBold />}
      colorScheme="error"
      onClick={api.openDialog}
      {...props}
    >
      {t('queue.clear')}
    </Button>
  );
});

ClearQueueButton.displayName = 'ClearQueueButton';

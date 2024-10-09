import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';

import { useClearQueue } from './ClearQueueConfirmationAlertDialog';

type Props = ButtonProps;

const ClearQueueButton = (props: Props) => {
  const { t } = useTranslation();
  const clearQueue = useClearQueue();

  return (
    <>
      <Button
        isDisabled={clearQueue.isDisabled}
        isLoading={clearQueue.isLoading}
        tooltip={t('queue.clearTooltip')}
        leftIcon={<PiTrashSimpleFill />}
        colorScheme="error"
        onClick={clearQueue.openDialog}
        data-testid={t('queue.clear')}
        {...props}
      >
        {t('queue.clear')}
      </Button>
    </>
  );
};

export default memo(ClearQueueButton);

import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button } from '@invoke-ai/ui-library';
import { useClearQueueConfirmationAlertDialog } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';

type Props = ButtonProps;

const ClearQueueButton = (props: Props) => {
  const { t } = useTranslation();
  const dialogState = useClearQueueConfirmationAlertDialog();
  const { isLoading, isDisabled } = useClearQueue();

  return (
    <>
      <Button
        isDisabled={isDisabled}
        isLoading={isLoading}
        tooltip={t('queue.clearTooltip')}
        leftIcon={<PiTrashSimpleFill />}
        colorScheme="error"
        onClick={dialogState.setTrue}
        data-testid={t('queue.clear')}
        {...props}
      >
        {t('queue.clear')}
      </Button>
    </>
  );
};

export default memo(ClearQueueButton);

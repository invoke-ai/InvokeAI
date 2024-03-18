import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button, useDisclosure } from '@invoke-ai/ui-library';
import ClearQueueConfirmationAlertDialog from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';

type Props = ButtonProps;

const ClearQueueButton = (props: Props) => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();
  const { isLoading, isDisabled } = useClearQueue();

  return (
    <>
      <Button
        isDisabled={isDisabled}
        isLoading={isLoading}
        tooltip={t('queue.clearTooltip')}
        leftIcon={<PiTrashSimpleFill />}
        colorScheme="error"
        onClick={disclosure.onOpen}
        data-testid={t('queue.clear')}
        {...props}
      >
        {t('queue.clear')}
      </Button>
      <ClearQueueConfirmationAlertDialog disclosure={disclosure} />
    </>
  );
};

export default memo(ClearQueueButton);

import { useDisclosure } from '@chakra-ui/react';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import type { InvIconButtonProps } from 'common/components/InvIconButton/types';
import ClearQueueConfirmationAlertDialog from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = Omit<InvIconButtonProps, 'aria-label'>;

const ClearQueueIconButton = (props: Props) => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();
  const { isLoading, isDisabled } = useClearQueue();

  return (
    <>
      <InvIconButton
        isDisabled={isDisabled}
        isLoading={isLoading}
        aria-label={t('queue.clear')}
        tooltip={t('queue.clearTooltip')}
        icon={<PiTrashSimpleBold size="16px" />}
        colorScheme="error"
        onClick={disclosure.onOpen}
        data-testid={t('queue.clear')}
        {...props}
      />
      <ClearQueueConfirmationAlertDialog disclosure={disclosure} />
    </>
  );
};

export default memo(ClearQueueIconButton);

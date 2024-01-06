import { useDisclosure } from '@chakra-ui/react';
import { InvButton } from 'common/components/InvButton/InvButton';
import type { InvButtonProps } from 'common/components/InvButton/types';
import ClearQueueConfirmationAlertDialog from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi'

type Props = InvButtonProps;

const ClearQueueButton = (props: Props) => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();
  const { isLoading, isDisabled } = useClearQueue();

  return (
    <>
      <InvButton
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
      </InvButton>
      <ClearQueueConfirmationAlertDialog disclosure={disclosure} />
    </>
  );
};

export default memo(ClearQueueButton);

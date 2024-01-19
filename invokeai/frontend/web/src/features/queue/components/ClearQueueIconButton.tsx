import { useDisclosure } from '@chakra-ui/react';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import type { InvIconButtonProps } from 'common/components/InvIconButton/types';
import ClearQueueConfirmationAlertDialog from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = Omit<InvIconButtonProps, 'aria-label'>;

const ClearQueueIconButton = ({
  onOpen,
  ...props
}: Props & { onOpen: () => void }) => {
  const { t } = useTranslation();
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
        onClick={onOpen}
        data-testid={t('queue.clear')}
        {...props}
      />
    </>
  );
};

const ClearSingleQueueItemIconButton = (props: Props) => {
  const { t } = useTranslation();
  const { cancelQueueItem, isLoading, isDisabled } =
    useCancelCurrentQueueItem();

  return (
    <>
      <InvIconButton
        isDisabled={isDisabled}
        isLoading={isLoading}
        aria-label={t('queue.cancelTooltip')}
        tooltip={t('queue.cancelTooltip')}
        icon={<PiTrashSimpleBold size="16px" />}
        colorScheme="error"
        onClick={cancelQueueItem}
        data-testid={t('queue.clear')}
        {...props}
      />
    </>
  );
};

export const ClearQueueButton = (props: Props) => {
  // Show the single item clear button when shift is pressed
  // Otherwise show the clear queue button
  const [showSingleItemClear, setShowSingleItemClear] = useState(true);
  useHotkeys('shift', () => setShowSingleItemClear(false), {
    keydown: true,
    keyup: false,
  });
  useHotkeys('shift', () => setShowSingleItemClear(true), {
    keydown: false,
    keyup: true,
  });

  const disclosure = useDisclosure();

  return (
    <>
      {showSingleItemClear ? (
        <ClearSingleQueueItemIconButton {...props} />
      ) : (
        <ClearQueueIconButton {...props} onOpen={disclosure.onOpen} />
      )}
      <ClearQueueConfirmationAlertDialog disclosure={disclosure} />
    </>
  );
};

export default ClearQueueButton;

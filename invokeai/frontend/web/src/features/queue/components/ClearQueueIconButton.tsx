import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton, useDisclosure, useShiftModifier } from '@invoke-ai/ui-library';
import ClearQueueConfirmationAlertDialog from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold, PiXBold } from 'react-icons/pi';

type ClearQueueButtonProps = Omit<IconButtonProps, 'aria-label'>;

type ClearQueueIconButtonProps = ClearQueueButtonProps & {
  onOpen: () => void;
};

export const ClearAllQueueIconButton = ({ onOpen, ...props }: ClearQueueIconButtonProps) => {
  const { t } = useTranslation();
  const { isLoading, isDisabled } = useClearQueue();

  return (
    <IconButton
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
  );
};

const ClearSingleQueueItemIconButton = (props: ClearQueueButtonProps) => {
  const { t } = useTranslation();
  const { cancelQueueItem, isLoading, isDisabled } = useCancelCurrentQueueItem();

  return (
    <IconButton
      isDisabled={isDisabled}
      isLoading={isLoading}
      aria-label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<PiXBold size="16px" />}
      colorScheme="error"
      onClick={cancelQueueItem}
      data-testid={t('queue.cancel')}
      {...props}
    />
  );
};

export const ClearQueueIconButton = (props: ClearQueueButtonProps) => {
  // Show the single item clear button when shift is pressed
  // Otherwise show the clear queue button
  const shift = useShiftModifier();
  const disclosure = useDisclosure();

  return (
    <>
      {shift ? (
        <ClearAllQueueIconButton {...props} onOpen={disclosure.onOpen} />
      ) : (
        <ClearSingleQueueItemIconButton {...props} />
      )}
      <ClearQueueConfirmationAlertDialog disclosure={disclosure} />
    </>
  );
};

export default ClearQueueIconButton;

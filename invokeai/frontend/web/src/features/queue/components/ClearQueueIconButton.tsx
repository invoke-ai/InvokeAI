import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import { useClearQueueConfirmationAlertDialog } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold, PiXBold } from 'react-icons/pi';

type ClearQueueButtonProps = Omit<IconButtonProps, 'aria-label'>;

export const ClearAllQueueIconButton = memo((props: ClearQueueButtonProps) => {
  const { t } = useTranslation();
  const dialogState = useClearQueueConfirmationAlertDialog();
  const { isLoading, isDisabled } = useClearQueue();

  return (
    <IconButton
      isDisabled={isDisabled}
      isLoading={isLoading}
      aria-label={t('queue.clear')}
      tooltip={t('queue.clearTooltip')}
      icon={<PiTrashSimpleBold size="16px" />}
      colorScheme="error"
      onClick={dialogState.setTrue}
      data-testid={t('queue.clear')}
      {...props}
    />
  );
});

ClearAllQueueIconButton.displayName = 'ClearAllQueueIconButton';

const ClearSingleQueueItemIconButton = memo((props: ClearQueueButtonProps) => {
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
});

ClearSingleQueueItemIconButton.displayName = 'ClearSingleQueueItemIconButton';

export const ClearQueueIconButton = memo((props: ClearQueueButtonProps) => {
  // Show the single item clear button when shift is pressed
  // Otherwise show the clear queue button
  const shift = useShiftModifier();

  if (shift) {
    return <ClearAllQueueIconButton {...props} />;
  }

  return <ClearSingleQueueItemIconButton {...props} />;
});

ClearQueueIconButton.displayName = 'ClearQueueIconButton';

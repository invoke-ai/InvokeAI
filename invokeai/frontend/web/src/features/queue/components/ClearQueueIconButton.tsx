import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold, PiXBold } from 'react-icons/pi';

import { useClearQueue } from './ClearQueueConfirmationAlertDialog';

export const ClearQueueIconButton = memo((_) => {
  const { t } = useTranslation();
  const clearQueue = useClearQueue();
  const cancelCurrentQueueItem = useCancelCurrentQueueItem();

  // Show the single item clear button when shift is pressed
  // Otherwise show the clear queue button
  const shift = useShiftModifier();

  return (
    <IconButton
      size="lg"
      isDisabled={shift ? clearQueue.isDisabled : cancelCurrentQueueItem.isDisabled}
      isLoading={shift ? clearQueue.isLoading : cancelCurrentQueueItem.isLoading}
      aria-label={shift ? t('queue.clear') : t('queue.cancel')}
      tooltip={shift ? t('queue.clearTooltip') : t('queue.cancelTooltip')}
      icon={shift ? <PiTrashSimpleBold /> : <PiXBold />}
      colorScheme="error"
      onClick={shift ? clearQueue.openDialog : cancelCurrentQueueItem.cancelQueueItem}
      data-testid={shift ? t('queue.clear') : t('queue.cancel')}
    />
  );
});

ClearQueueIconButton.displayName = 'ClearQueueIconButton';

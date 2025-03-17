import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import { useCancelAllExceptCurrentQueueItemDialog } from 'features/queue/components/CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold, PiXBold, PiXCircle } from 'react-icons/pi';

import { useClearQueueDialog } from './ClearQueueConfirmationAlertDialog';

export const ClearQueueIconButton = memo(() => {
  const isCancelAndClearAllEnabled = useFeatureStatus('cancelAndClearAll');
  const shift = useShiftModifier();

  if (!shift) {
    // Shift is not pressed - show cancel current
    return <CancelCurrentIconButton />;
  }

  if (isCancelAndClearAllEnabled) {
    // Shift is pressed and cancel and clear all is enabled - show cancel and clear all
    return <CancelAndClearAllIconButton />;
  }

  // Shift is pressed and cancel and clear all is disabled - show cancel all except current
  return <CancelAllExceptCurrentIconButton />;
});

ClearQueueIconButton.displayName = 'ClearQueueIconButton';

const CancelCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const cancelCurrentQueueItem = useCancelCurrentQueueItem();

  return (
    <IconButton
      size="lg"
      isDisabled={cancelCurrentQueueItem.isDisabled}
      isLoading={cancelCurrentQueueItem.isLoading}
      aria-label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<PiXBold />}
      colorScheme="error"
      onClick={cancelCurrentQueueItem.cancelQueueItem}
    />
  );
});

CancelCurrentIconButton.displayName = 'CancelCurrentIconButton';

const CancelAndClearAllIconButton = memo(() => {
  const { t } = useTranslation();
  const clearQueue = useClearQueueDialog();

  return (
    <IconButton
      size="lg"
      isDisabled={clearQueue.isDisabled}
      isLoading={clearQueue.isLoading}
      aria-label={t('queue.clear')}
      tooltip={t('queue.clearTooltip')}
      icon={<PiTrashSimpleBold />}
      colorScheme="error"
      onClick={clearQueue.openDialog}
    />
  );
});

CancelAndClearAllIconButton.displayName = 'CancelAndClearAllIconButton';

const CancelAllExceptCurrentIconButton = memo(() => {
  const { t } = useTranslation();
  const cancelAllExceptCurrent = useCancelAllExceptCurrentQueueItemDialog();

  return (
    <IconButton
      size="lg"
      isDisabled={cancelAllExceptCurrent.isDisabled}
      isLoading={cancelAllExceptCurrent.isLoading}
      aria-label={t('queue.clear')}
      tooltip={t('queue.cancelAllExceptCurrentTooltip')}
      icon={<PiXCircle />}
      colorScheme="error"
      onClick={cancelAllExceptCurrent.openDialog}
    />
  );
});

CancelAllExceptCurrentIconButton.displayName = 'CancelAllExceptCurrentIconButton';

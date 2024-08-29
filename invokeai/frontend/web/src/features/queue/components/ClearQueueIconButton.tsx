import { IconButton, useShiftModifier } from '@invoke-ai/ui-library';
import { QueueCountBadge } from 'features/queue/components/QueueCountBadge';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { useClearQueue } from 'features/queue/hooks/useClearQueue';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold, PiXBold } from 'react-icons/pi';

export const ClearQueueIconButton = memo((_) => {
  const ref = useRef<HTMLDivElement>(null);
  const { t } = useTranslation();
  const clearQueue = useClearQueue();
  const cancelCurrentQueueItem = useCancelCurrentQueueItem();

  // Show the single item clear button when shift is pressed
  // Otherwise show the clear queue button
  const shift = useShiftModifier();

  return (
    <>
      <IconButton
        ref={ref}
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
      {/* The badge is dynamically positioned, needs a ref to the target element */}
      <QueueCountBadge targetRef={ref} />
    </>
  );
});

ClearQueueIconButton.displayName = 'ClearQueueIconButton';

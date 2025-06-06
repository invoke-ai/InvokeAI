import { Flex, Spacer, useShiftModifier } from '@invoke-ai/ui-library';
import { DeleteAllExceptCurrentQueueItemConfirmationAlertDialog } from 'features/queue/components/DeleteAllExceptCurrentQueueItemConfirmationAlertDialog';
import { DeleteCurrentQueueItemIconButton } from 'features/queue/components/DeleteCurrentQueueItemIconButton';
import { QueueActionsMenuButton } from 'features/queue/components/QueueActionsMenuButton';
import ProgressBar from 'features/system/components/ProgressBar';
import { memo } from 'react';

import { InvokeButton } from './InvokeQueueBackButton';

const QueueControls = () => {
  return (
    <Flex w="full" position="relative" borderRadius="base" gap={2} flexDir="column">
      <Flex gap={2}>
        <InvokeButton />
        <Spacer />
        <QueueActionsMenuButton />
        <DeleteIconButton />
      </Flex>
      <ProgressBar />
    </Flex>
  );
};

export default memo(QueueControls);

export const DeleteIconButton = memo(() => {
  const shift = useShiftModifier();

  if (!shift) {
    return <DeleteCurrentQueueItemIconButton />;
  }

  return <DeleteAllExceptCurrentQueueItemConfirmationAlertDialog />;
});

DeleteIconButton.displayName = 'DeleteIconButton';

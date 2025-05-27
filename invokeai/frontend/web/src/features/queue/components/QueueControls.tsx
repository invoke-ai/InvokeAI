import { Flex, Spacer } from '@invoke-ai/ui-library';
import { ClearQueueIconButton } from 'features/queue/components/ClearQueueIconButton';
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
        <ClearQueueIconButton />
      </Flex>
      <ProgressBar />
    </Flex>
  );
};

export default memo(QueueControls);

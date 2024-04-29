import { ButtonGroup, Flex, Spacer } from '@invoke-ai/ui-library';
import { ClearQueueIconButton } from 'features/queue/components/ClearQueueIconButton';
import QueueFrontButton from 'features/queue/components/QueueFrontButton';
import ProgressBar from 'features/system/components/ProgressBar';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';

import { InvokeQueueBackButton } from './InvokeQueueBackButton';
import { QueueActionsMenuButton } from './QueueActionsMenuButton';

const QueueControls = () => {
  const isPrependEnabled = useFeatureStatus('prependQueue');
  return (
    <Flex w="full" position="relative" borderRadius="base" gap={2} pt={2} flexDir="column">
      <ButtonGroup size="lg" isAttached={false}>
        {isPrependEnabled && <QueueFrontButton />}
        <InvokeQueueBackButton />
        <Spacer />
        <QueueActionsMenuButton />
        {/* <CancelCurrentQueueItemButton asIconButton />
        {isResumeEnabled && <ResumeProcessorButton asIconButton />}
        {isPauseEnabled && <PauseProcessorButton asIconButton />} */}
        <ClearQueueIconButton />
      </ButtonGroup>
      <ProgressBar />
    </Flex>
  );
};

export default memo(QueueControls);

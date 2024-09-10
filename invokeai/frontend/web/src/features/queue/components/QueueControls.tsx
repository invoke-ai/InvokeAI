import { Flex, Spacer } from '@invoke-ai/ui-library';
import { ClearQueueIconButton } from 'features/queue/components/ClearQueueIconButton';
import QueueFrontButton from 'features/queue/components/QueueFrontButton';
import ProgressBar from 'features/system/components/ProgressBar';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';

import { InvokeQueueBackButton } from './InvokeQueueBackButton';

const QueueControls = () => {
  const isPrependEnabled = useFeatureStatus('prependQueue');

  return (
    <Flex w="full" position="relative" borderRadius="base" gap={2} flexDir="column">
      <Flex gap={2}>
        {isPrependEnabled && <QueueFrontButton />}
        <InvokeQueueBackButton />
        <Spacer />
        <ClearQueueIconButton />
      </Flex>
      <ProgressBar />
    </Flex>
  );
};

export default memo(QueueControls);

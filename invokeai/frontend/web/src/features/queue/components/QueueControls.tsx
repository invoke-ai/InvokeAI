import { Flex, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ClearQueueIconButton } from 'features/queue/components/ClearQueueIconButton';
import QueueFrontButton from 'features/queue/components/QueueFrontButton';
import { SendToToggle } from 'features/queue/components/SendToToggle';
import ProgressBar from 'features/system/components/ProgressBar';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

import { InvokeQueueBackButton } from './InvokeQueueBackButton';

const QueueControls = () => {
  const isPrependEnabled = useFeatureStatus('prependQueue');
  const tab = useAppSelector(selectActiveTab);

  return (
    <Flex w="full" position="relative" borderRadius="base" gap={2} flexDir="column">
      <Flex gap={2}>
        {isPrependEnabled && <QueueFrontButton />}
        <InvokeQueueBackButton />
        <Spacer />
        {tab === 'generation' && <SendToToggle />}
        <ClearQueueIconButton />
      </Flex>
      <ProgressBar />
    </Flex>
  );
};

export default memo(QueueControls);

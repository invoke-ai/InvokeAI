import { Flex } from '@chakra-ui/react';
import { InvButtonGroup } from 'common/components/InvButtonGroup/InvButtonGroup';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';

import ClearQueueButton from './ClearQueueButton';
import PauseProcessorButton from './PauseProcessorButton';
import PruneQueueButton from './PruneQueueButton';
import ResumeProcessorButton from './ResumeProcessorButton';

const QueueTabQueueControls = () => {
  const isPauseEnabled = useFeatureStatus('pauseQueue').isFeatureEnabled;
  const isResumeEnabled = useFeatureStatus('resumeQueue').isFeatureEnabled;
  return (
    <Flex layerStyle="first" borderRadius="base" p={2} gap={2}>
      {isPauseEnabled || isResumeEnabled ? (
        <InvButtonGroup w={28} orientation="vertical" size="sm">
          {isResumeEnabled ? <ResumeProcessorButton /> : <></>}
          {isPauseEnabled ? <PauseProcessorButton /> : <></>}
        </InvButtonGroup>
      ) : (
        <></>
      )}
      <InvButtonGroup w={28} orientation="vertical" size="sm">
        <PruneQueueButton />
        <ClearQueueButton />
      </InvButtonGroup>
    </Flex>
  );
};

export default memo(QueueTabQueueControls);

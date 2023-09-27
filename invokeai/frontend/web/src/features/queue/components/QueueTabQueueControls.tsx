import { ButtonGroup, Flex } from '@chakra-ui/react';
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
    <Flex layerStyle="second" borderRadius="base" p={2} gap={2}>
      {isPauseEnabled || isResumeEnabled ? (
        <ButtonGroup w={28} orientation="vertical" isAttached size="sm">
          {isResumeEnabled ? <ResumeProcessorButton /> : <></>}
          {isPauseEnabled ? <PauseProcessorButton /> : <></>}
        </ButtonGroup>
      ) : (
        <></>
      )}
      <ButtonGroup w={28} orientation="vertical" isAttached size="sm">
        <PruneQueueButton />
        <ClearQueueButton />
      </ButtonGroup>
    </Flex>
  );
};

export default memo(QueueTabQueueControls);

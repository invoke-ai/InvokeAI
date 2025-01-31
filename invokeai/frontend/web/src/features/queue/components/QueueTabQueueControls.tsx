/* eslint-disable i18next/no-literal-string */
import { ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';

import ClearModelCacheButton from './ClearModelCacheButton';
import ClearQueueButton from './ClearQueueButton';
import PauseProcessorButton from './PauseProcessorButton';
import PruneQueueButton from './PruneQueueButton';
import ResumeProcessorButton from './ResumeProcessorButton';

const QueueTabQueueControls = () => {
  const isPauseEnabled = useFeatureStatus('pauseQueue');
  const isResumeEnabled = useFeatureStatus('resumeQueue');
  return (
    <Flex flexDir="column" layerStyle="first" borderRadius="base" p={2} gap={2}>
      <Flex gap={2}>
        {isPauseEnabled || isResumeEnabled ? (
          <ButtonGroup w={28} orientation="vertical" size="sm">
            {isResumeEnabled ? <ResumeProcessorButton /> : <></>}
            {isPauseEnabled ? <PauseProcessorButton /> : <></>}
          </ButtonGroup>
        ) : (
          <></>
        )}
        <ButtonGroup w={28} orientation="vertical" size="sm">
          <PruneQueueButton />
          <ClearQueueButton />
        </ButtonGroup>
      </Flex>
      <ClearModelCacheButton />
    </Flex>
  );
};

export default memo(QueueTabQueueControls);

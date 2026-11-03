import { ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';

import ClearModelCacheButton from './ClearModelCacheButton';
import { ClearQueueButton } from './ClearQueueButton';
import PauseProcessorButton from './PauseProcessorButton';
import PruneQueueButton from './PruneQueueButton';
import ResumeProcessorButton from './ResumeProcessorButton';

const QueueTabQueueControls = () => {
  return (
    <Flex flexDir="column" layerStyle="first" borderRadius="base" p={2} gap={2}>
      <Flex gap={2}>
        <ButtonGroup orientation="vertical" size="sm">
          <ResumeProcessorButton />
          <PauseProcessorButton />
        </ButtonGroup>
        <ButtonGroup orientation="vertical" size="sm">
          <PruneQueueButton />
          <ClearQueueButton />
        </ButtonGroup>
      </Flex>
      <ClearModelCacheButton />
    </Flex>
  );
};

export default memo(QueueTabQueueControls);

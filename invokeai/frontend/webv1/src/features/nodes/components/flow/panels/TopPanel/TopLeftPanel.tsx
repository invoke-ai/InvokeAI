import { Flex } from '@invoke-ai/ui-library';
import AddNodeButton from 'features/nodes/components/flow/panels/TopPanel/AddNodeButton';
import UpdateNodesButton from 'features/nodes/components/flow/panels/TopPanel/UpdateNodesButton';
import { memo } from 'react';

export const TopLeftPanel = memo(() => {
  return (
    <Flex gap={2} top={2} left={2} position="absolute" alignItems="flex-start" pointerEvents="none">
      <Flex gap="2">
        <AddNodeButton />
        <UpdateNodesButton />
      </Flex>
    </Flex>
  );
});

TopLeftPanel.displayName = 'TopLeftPanel';

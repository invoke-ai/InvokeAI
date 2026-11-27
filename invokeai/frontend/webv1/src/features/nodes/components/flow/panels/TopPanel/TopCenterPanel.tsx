import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { WorkflowName } from 'features/nodes/components/sidePanel/WorkflowName';
import { selectWorkflowName } from 'features/nodes/store/selectors';
import { memo } from 'react';

export const TopCenterPanel = memo(() => {
  const name = useAppSelector(selectWorkflowName);
  return (
    <Flex gap={2} top={2} left="50%" transform="translateX(-50%)" position="absolute" pointerEvents="none">
      {!!name.length && <WorkflowName />}
    </Flex>
  );
});
TopCenterPanel.displayName = 'TopCenterPanel';

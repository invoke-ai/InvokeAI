import 'reactflow/dist/style.css';

import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { LinearViewLeftPanelContent } from 'features/nodes/components/sidePanel/LinearViewLeftPanelContent';
import { useWorkflowListMenu } from 'features/nodes/store/workflowListMenu';
import { selectWorkflowMode } from 'features/nodes/store/workflowSlice';
import { memo } from 'react';

import { SimpleViewLeftPanelContent } from './viewMode/SimpleViewLeftPanelContent';
import { WorkflowListMenu } from './WorkflowListMenu/WorkflowListMenu';
import { WorkflowListMenuTrigger } from './WorkflowListMenu/WorkflowListMenuTrigger';

const WorkflowsTabLeftPanel = () => {
  const mode = useAppSelector(selectWorkflowMode);
  const workflowListMenu = useWorkflowListMenu();

  return (
    <Flex w="full" h="full" gap={2} flexDir="column">
      <WorkflowListMenuTrigger />
      <Flex w="full" h="full" position="relative">
        <Box position="absolute" top={0} left={0} right={0} bottom={0}>
          {workflowListMenu.isOpen && <WorkflowListMenu />}
          {mode === 'view' && <SimpleViewLeftPanelContent />}
          {mode === 'edit' && <LinearViewLeftPanelContent />}
        </Box>
      </Flex>
    </Flex>
  );
};

export default memo(WorkflowsTabLeftPanel);

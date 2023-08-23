import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import { Panel } from 'reactflow';
import WorkflowEditorControls from './WorkflowEditorControls';

const TopCenterPanel = () => {
  return (
    <Panel position="top-center">
      <Flex gap={2}>
        <WorkflowEditorControls />
      </Flex>
    </Panel>
  );
};

export default memo(TopCenterPanel);

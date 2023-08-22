import { HStack } from '@chakra-ui/react';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import { memo } from 'react';
import { Panel } from 'reactflow';
import NodeEditorSettings from './NodeEditorSettings';
import ClearGraphButton from './ClearGraphButton';
import NodeInvokeButton from './NodeInvokeButton';
import ReloadSchemaButton from './ReloadSchemaButton';
import LoadWorkflowButton from './LoadWorkflowButton';

const TopCenterPanel = () => {
  return (
    <Panel position="top-center">
      <HStack>
        <NodeInvokeButton />
        <CancelButton />
        <ReloadSchemaButton />
        <ClearGraphButton />
        <NodeEditorSettings />
        <LoadWorkflowButton />
      </HStack>
    </Panel>
  );
};

export default memo(TopCenterPanel);

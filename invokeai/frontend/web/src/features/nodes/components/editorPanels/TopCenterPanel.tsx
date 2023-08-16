import { HStack } from '@chakra-ui/react';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import { memo } from 'react';
import { Panel } from 'reactflow';
import NodeEditorSettings from '../NodeEditorSettings';
import ClearGraphButton from '../ui/ClearGraphButton';
import NodeInvokeButton from '../ui/NodeInvokeButton';
import ReloadSchemaButton from '../ui/ReloadSchemaButton';

const TopCenterPanel = () => {
  return (
    <Panel position="top-center">
      <HStack>
        <NodeInvokeButton />
        <CancelButton />
        <ReloadSchemaButton />
        <ClearGraphButton />
        <NodeEditorSettings />
      </HStack>
    </Panel>
  );
};

export default memo(TopCenterPanel);

import { HStack } from '@chakra-ui/react';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import { memo } from 'react';
import { Panel } from 'reactflow';
import NodeEditorSettings from './NodeEditorSettings';
import ClearGraphButton from './ClearGraphButton';
import NodeInvokeButton from './NodeInvokeButton';
import ReloadSchemaButton from './ReloadSchemaButton';

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

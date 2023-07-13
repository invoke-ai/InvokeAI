import { HStack } from '@chakra-ui/react';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import { memo } from 'react';
import { Panel } from 'reactflow';
import LoadNodesButton from '../ui/LoadNodesButton';
import NodeInvokeButton from '../ui/NodeInvokeButton';
import ReloadSchemaButton from '../ui/ReloadSchemaButton';
import SaveNodesButton from '../ui/SaveNodesButton';

const TopCenterPanel = () => {
  return (
    <Panel position="top-center">
      <HStack>
        <NodeInvokeButton />
        <CancelButton />
        <ReloadSchemaButton />
        <SaveNodesButton />
        <LoadNodesButton />
      </HStack>
    </Panel>
  );
};

export default memo(TopCenterPanel);

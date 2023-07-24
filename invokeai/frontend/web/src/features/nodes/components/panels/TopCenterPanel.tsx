import { HStack } from '@chakra-ui/react';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import { memo } from 'react';
import { Panel } from 'reactflow';
import ClearGraphButton from '../ui/ClearGraphButton';
import LoadGraphButton from '../ui/LoadGraphButton';
import NodeInvokeButton from '../ui/NodeInvokeButton';
import ReloadSchemaButton from '../ui/ReloadSchemaButton';
import SaveGraphButton from '../ui/SaveGraphButton';

const TopCenterPanel = () => {
  return (
    <Panel position="top-center">
      <HStack>
        <NodeInvokeButton />
        <CancelButton />
        <ReloadSchemaButton />
        <SaveGraphButton />
        <LoadGraphButton />
        <ClearGraphButton />
      </HStack>
    </Panel>
  );
};

export default memo(TopCenterPanel);

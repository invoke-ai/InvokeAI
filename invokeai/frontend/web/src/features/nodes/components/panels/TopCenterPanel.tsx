import { HStack } from '@chakra-ui/react';
import { memo } from 'react';
import { Panel } from 'reactflow';

import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import NodeInvokeButton from '../ui/NodeInvokeButton';
import ReloadSchemaButton from '../ui/ReloadSchemaButton';

const TopCenterPanel = () => {
  return (
    <Panel position="top-center">
      <HStack>
        <NodeInvokeButton />
        <CancelButton />
        <ReloadSchemaButton />
      </HStack>
    </Panel>
  );
};

export default memo(TopCenterPanel);

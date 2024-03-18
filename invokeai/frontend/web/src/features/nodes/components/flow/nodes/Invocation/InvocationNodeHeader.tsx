import { Flex } from '@invoke-ai/ui-library';
import NodeCollapseButton from 'features/nodes/components/flow/nodes/common/NodeCollapseButton';
import NodeTitle from 'features/nodes/components/flow/nodes/common/NodeTitle';
import InvocationNodeClassificationIcon from 'features/nodes/components/flow/nodes/Invocation/InvocationNodeClassificationIcon';
import { memo } from 'react';

import InvocationNodeCollapsedHandles from './InvocationNodeCollapsedHandles';
import InvocationNodeInfoIcon from './InvocationNodeInfoIcon';
import InvocationNodeStatusIndicator from './InvocationNodeStatusIndicator';

type Props = {
  nodeId: string;
  isOpen: boolean;
  label: string;
  type: string;
  selected: boolean;
};

const InvocationNodeHeader = ({ nodeId, isOpen }: Props) => {
  return (
    <Flex
      layerStyle="nodeHeader"
      borderTopRadius="base"
      borderBottomRadius={isOpen ? 0 : 'base'}
      alignItems="center"
      justifyContent="space-between"
      h={8}
      textAlign="center"
      color="base.200"
    >
      <NodeCollapseButton nodeId={nodeId} isOpen={isOpen} />
      <InvocationNodeClassificationIcon nodeId={nodeId} />
      <NodeTitle nodeId={nodeId} />
      <Flex alignItems="center">
        <InvocationNodeStatusIndicator nodeId={nodeId} />
        <InvocationNodeInfoIcon nodeId={nodeId} />
      </Flex>
      {!isOpen && <InvocationNodeCollapsedHandles nodeId={nodeId} />}
    </Flex>
  );
};

export default memo(InvocationNodeHeader);

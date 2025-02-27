import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import NodeCollapseButton from 'features/nodes/components/flow/nodes/common/NodeCollapseButton';
import NodeTitle from 'features/nodes/components/flow/nodes/common/NodeTitle';
import InvocationNodeClassificationIcon from 'features/nodes/components/flow/nodes/Invocation/InvocationNodeClassificationIcon';
import { useNodeIsInvalid } from 'features/nodes/hooks/useNodeIsInvalid';
import { memo } from 'react';

import InvocationNodeCollapsedHandles from './InvocationNodeCollapsedHandles';
import { InvocationNodeInfoIcon } from './InvocationNodeInfoIcon';
import InvocationNodeStatusIndicator from './InvocationNodeStatusIndicator';

type Props = {
  nodeId: string;
  isOpen: boolean;
};

const sx: SystemStyleObject = {
  borderTopRadius: 'base',
  alignItems: 'center',
  justifyContent: 'space-between',
  h: 8,
  textAlign: 'center',
  color: 'base.200',
  borderBottomRadius: 'base',
  '&[data-is-open="true"]': {
    borderBottomRadius: 0,
  },
  '&[data-is-invalid="true"]': {
    color: 'error.300',
  },
};

const InvocationNodeHeader = ({ nodeId, isOpen }: Props) => {
  const isInvalid = useNodeIsInvalid(nodeId);

  return (
    <Flex layerStyle="nodeHeader" sx={sx} data-is-open={isOpen} data-is-invalid={isInvalid}>
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

import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import InvocationNodeTitle from 'features/nodes/components/flow/nodes/common/InvocationNodeTitle';
import NodeCollapseButton from 'features/nodes/components/flow/nodes/common/NodeCollapseButton';
import InvocationNodeClassificationIcon from 'features/nodes/components/flow/nodes/Invocation/InvocationNodeClassificationIcon';
import { useNodeHasErrors } from 'features/nodes/hooks/useNodeIsInvalid';
import { memo } from 'react';

import InvocationNodeCollapsedHandles from './InvocationNodeCollapsedHandles';
import { InvocationNodeHelpButton } from './InvocationNodeHelpButton';
import { InvocationNodeInfoIcon } from './InvocationNodeInfoIcon';
import InvocationNodeStatusIndicator from './InvocationNodeStatusIndicator';

type Props = {
  nodeId: string;
  isOpen: boolean;
};

const sx: SystemStyleObject = {
  bg: 'var(--header-bg-color)',
  borderTopRadius: 'base',
  alignItems: 'center',
  justifyContent: 'space-between',
  h: 8,
  textAlign: 'center',
  borderBottomRadius: 'base',
  '&[data-is-open="true"]': {
    borderBottomRadius: 0,
  },
};

const InvocationNodeHeader = ({ nodeId, isOpen }: Props) => {
  const isInvalid = useNodeHasErrors();

  return (
    <Flex sx={sx} data-is-open={isOpen} data-is-invalid={isInvalid}>
      <NodeCollapseButton nodeId={nodeId} isOpen={isOpen} />
      <InvocationNodeClassificationIcon nodeId={nodeId} />
      <InvocationNodeTitle nodeId={nodeId} />
      <Flex alignItems="center">
        <InvocationNodeStatusIndicator nodeId={nodeId} />
        <InvocationNodeHelpButton nodeId={nodeId} />
        <InvocationNodeInfoIcon nodeId={nodeId} />
      </Flex>
      {!isOpen && <InvocationNodeCollapsedHandles nodeId={nodeId} />}
    </Flex>
  );
};

export default memo(InvocationNodeHeader);

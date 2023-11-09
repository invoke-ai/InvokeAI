import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import NodeCollapseButton from '../common/NodeCollapseButton';
import NodeTitle from '../common/NodeTitle';
import InvocationNodeCollapsedHandles from './InvocationNodeCollapsedHandles';
import InvocationNodeNotes from './InvocationNodeNotes';
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
      sx={{
        borderTopRadius: 'base',
        borderBottomRadius: isOpen ? 0 : 'base',
        alignItems: 'center',
        justifyContent: 'space-between',
        h: 8,
        textAlign: 'center',
        fontWeight: 500,
        color: 'base.700',
        _dark: { color: 'base.200' },
      }}
    >
      <NodeCollapseButton nodeId={nodeId} isOpen={isOpen} />
      <NodeTitle nodeId={nodeId} />
      <Flex alignItems="center">
        <InvocationNodeStatusIndicator nodeId={nodeId} />
        <InvocationNodeNotes nodeId={nodeId} />
      </Flex>
      {!isOpen && <InvocationNodeCollapsedHandles nodeId={nodeId} />}
    </Flex>
  );
};

export default memo(InvocationNodeHeader);

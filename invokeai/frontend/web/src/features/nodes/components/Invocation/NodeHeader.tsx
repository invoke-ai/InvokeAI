import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import NodeCollapseButton from '../Invocation/NodeCollapseButton';
import NodeCollapsedHandles from '../Invocation/NodeCollapsedHandles';
import NodeNotesEdit from '../Invocation/NodeNotesEdit';
import NodeStatusIndicator from '../Invocation/NodeStatusIndicator';
import NodeTitle from '../Invocation/NodeTitle';

type Props = {
  nodeId: string;
  isOpen: boolean;
  label: string;
  type: string;
  selected: boolean;
};

const NodeHeader = ({ nodeId, isOpen }: Props) => {
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
        fontWeight: 600,
        color: 'base.700',
        _dark: { color: 'base.200' },
      }}
    >
      <NodeCollapseButton nodeId={nodeId} isOpen={isOpen} />
      <NodeTitle nodeId={nodeId} />
      <Flex alignItems="center">
        <NodeStatusIndicator nodeId={nodeId} />
        <NodeNotesEdit nodeId={nodeId} />
      </Flex>
      {!isOpen && <NodeCollapsedHandles nodeId={nodeId} />}
    </Flex>
  );
};

export default memo(NodeHeader);

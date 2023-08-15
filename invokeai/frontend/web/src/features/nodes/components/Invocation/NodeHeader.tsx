import { Flex } from '@chakra-ui/react';
import {
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { NodeProps } from 'reactflow';
import NodeCollapseButton from '../Invocation/NodeCollapseButton';
import NodeCollapsedHandles from '../Invocation/NodeCollapsedHandles';
import NodeNotesEdit from '../Invocation/NodeNotesEdit';
import NodeStatusIndicator from '../Invocation/NodeStatusIndicator';
import NodeTitle from '../Invocation/NodeTitle';

type Props = {
  nodeProps: NodeProps<InvocationNodeData>;
  nodeTemplate: InvocationTemplate;
};

const NodeHeader = (props: Props) => {
  const { nodeProps, nodeTemplate } = props;
  const { isOpen } = nodeProps.data;

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
      <NodeCollapseButton nodeProps={nodeProps} />
      <NodeTitle nodeData={nodeProps.data} title={nodeTemplate.title} />
      <Flex alignItems="center">
        <NodeStatusIndicator nodeProps={nodeProps} />
        <NodeNotesEdit nodeProps={nodeProps} nodeTemplate={nodeTemplate} />
      </Flex>
      {!isOpen && (
        <NodeCollapsedHandles
          nodeProps={nodeProps}
          nodeTemplate={nodeTemplate}
        />
      )}
    </Flex>
  );
};

export default memo(NodeHeader);

import { Box, Flex } from '@chakra-ui/react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { memo } from 'react';

import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from '../../../workflows/connectors';
import type { ConnectorFlowNode as ConnectorFlowNodeType } from './flowAdapters';

const HANDLE_SIZE = 10;

const handleStyle: React.CSSProperties = {
  background: 'var(--wb-flow-grid)',
  border: '1px solid var(--chakra-colors-bg)',
  height: HANDLE_SIZE,
  width: HANDLE_SIZE,
};

const ConnectorFlowNodeComponent = ({ data, selected }: NodeProps<ConnectorFlowNodeType>) => {
  const node = data.documentNode;

  return (
    <Flex align="center" data-connector-node-id={node.id} justify="center" position="relative" title="Connector">
      <Handle
        id={CONNECTOR_INPUT_HANDLE}
        position={Position.Left}
        style={{ ...handleStyle, left: -HANDLE_SIZE / 2 }}
        type="target"
      />
      <Box
        bg="bg"
        borderColor={selected ? 'accent.solid' : 'border.emphasized'}
        borderWidth="1px"
        h="1rem"
        w="2.5rem"
        rounded="full"
        shadow={selected ? 'md' : 'sm'}
      />
      <Handle
        id={CONNECTOR_OUTPUT_HANDLE}
        position={Position.Right}
        style={{ ...handleStyle, right: -HANDLE_SIZE / 2 }}
        type="source"
      />
    </Flex>
  );
};

export const ConnectorFlowNode = memo(ConnectorFlowNodeComponent);

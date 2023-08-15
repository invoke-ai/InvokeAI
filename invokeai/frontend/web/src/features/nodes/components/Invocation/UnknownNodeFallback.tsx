import { Box, Flex, Text } from '@chakra-ui/react';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { InvocationNodeData } from 'features/nodes/types/types';
import { memo } from 'react';
import { NodeProps } from 'reactflow';
import NodeCollapseButton from '../Invocation/NodeCollapseButton';
import NodeWrapper from '../Invocation/NodeWrapper';

type Props = {
  nodeProps: NodeProps<InvocationNodeData>;
};

const UnknownNodeFallback = ({ nodeProps }: Props) => {
  const { data } = nodeProps;
  const { isOpen, label, type } = data;
  return (
    <NodeWrapper nodeProps={nodeProps}>
      <Flex
        className={DRAG_HANDLE_CLASSNAME}
        layerStyle="nodeHeader"
        sx={{
          borderTopRadius: 'base',
          borderBottomRadius: isOpen ? 0 : 'base',
          alignItems: 'center',
          h: 8,
          fontWeight: 600,
          fontSize: 'sm',
        }}
      >
        <NodeCollapseButton nodeProps={nodeProps} />
        <Text
          sx={{
            w: 'full',
            textAlign: 'center',
            pe: 8,
            color: 'error.500',
            _dark: { color: 'error.300' },
          }}
        >
          {label ? `${label} (${type})` : type}
        </Text>
      </Flex>
      {isOpen && (
        <Flex
          layerStyle="nodeBody"
          sx={{
            userSelect: 'auto',
            flexDirection: 'column',
            w: 'full',
            h: 'full',
            p: 4,
            gap: 1,
            borderBottomRadius: 'base',
            fontSize: 'sm',
          }}
        >
          <Box>
            <Text as="span">Unknown node type: </Text>
            <Text as="span" fontWeight={600}>
              {type}
            </Text>
          </Box>
        </Flex>
      )}
    </NodeWrapper>
  );
};

export default memo(UnknownNodeFallback);

import { Box, Flex, Text } from '@chakra-ui/react';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { memo } from 'react';
import NodeCollapseButton from '../common/NodeCollapseButton';
import NodeWrapper from '../common/NodeWrapper';

type Props = {
  nodeId: string;
  isOpen: boolean;
  label: string;
  type: string;
  selected: boolean;
};

const InvocationNodeUnknownFallback = ({
  nodeId,
  isOpen,
  label,
  type,
  selected,
}: Props) => {
  return (
    <NodeWrapper nodeId={nodeId} selected={selected}>
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
        <NodeCollapseButton nodeId={nodeId} isOpen={isOpen} />
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

export default memo(InvocationNodeUnknownFallback);

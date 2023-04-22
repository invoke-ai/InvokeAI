import { NodeProps, NodeResizeControl } from 'reactflow';
import { Box, Flex, Icon } from '@chakra-ui/react';
import { FaExclamationCircle } from 'react-icons/fa';
import { InvocationValue } from '../types/types';

import { memo, useRef } from 'react';
import { useGetInvocationTemplate } from '../hooks/useInvocationTemplate';
import IAINodeOutputs from './IAINode/IAINodeOutputs';
import IAINodeInputs from './IAINode/IAINodeInputs';
import IAINodeHeader from './IAINode/IAINodeHeader';
import { IoResize } from 'react-icons/io5';
import IAINodeResizer from './IAINode/IAINodeResizer';

export const InvocationComponent = memo((props: NodeProps<InvocationValue>) => {
  const { id: nodeId, data, selected } = props;
  const { type, inputs, outputs } = data;

  const getInvocationTemplate = useGetInvocationTemplate();
  // TODO: determine if a field/handle is connected and disable the input if so

  const template = useRef(getInvocationTemplate(type));

  if (!template.current) {
    return (
      <Box
        sx={{
          padding: 4,
          bg: 'base.800',
          borderRadius: 'md',
          boxShadow: 'dark-lg',
          borderWidth: 2,
          borderColor: selected ? 'base.400' : 'transparent',
        }}
      >
        <Flex sx={{ alignItems: 'center', justifyContent: 'center' }}>
          <Icon color="base.400" boxSize={32} as={FaExclamationCircle}></Icon>
          <IAINodeResizer />
        </Flex>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        bg: 'base.800',
        borderRadius: 'md',
        boxShadow: 'dark-lg',
        borderWidth: 2,
        borderColor: selected ? 'base.400' : 'transparent',
      }}
    >
      <Flex flexDirection="column" gap={2}>
        <IAINodeHeader nodeId={nodeId} template={template} />
        <IAINodeOutputs nodeId={nodeId} outputs={outputs} template={template} />
        <IAINodeInputs nodeId={nodeId} inputs={inputs} template={template} />
        <IAINodeResizer />
      </Flex>
    </Box>
  );
});

InvocationComponent.displayName = 'InvocationComponent';

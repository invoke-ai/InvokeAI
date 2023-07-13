import { Flex, Icon } from '@chakra-ui/react';
import { FaExclamationCircle } from 'react-icons/fa';
import { NodeProps } from 'reactflow';
import { InvocationValue } from '../types/types';

import { useAppSelector } from 'app/store/storeHooks';
import { memo, useMemo } from 'react';
import { makeTemplateSelector } from '../store/util/makeTemplateSelector';
import IAINodeHeader from './IAINode/IAINodeHeader';
import IAINodeInputs from './IAINode/IAINodeInputs';
import IAINodeOutputs from './IAINode/IAINodeOutputs';
import IAINodeResizer from './IAINode/IAINodeResizer';
import NodeWrapper from './NodeWrapper';

export const InvocationComponent = memo((props: NodeProps<InvocationValue>) => {
  const { id: nodeId, data, selected } = props;
  const { type, inputs, outputs } = data;

  const templateSelector = useMemo(() => makeTemplateSelector(type), [type]);

  const template = useAppSelector(templateSelector);

  if (!template) {
    return (
      <NodeWrapper selected={selected}>
        <Flex sx={{ alignItems: 'center', justifyContent: 'center' }}>
          <Icon
            as={FaExclamationCircle}
            sx={{
              boxSize: 32,
              color: 'base.600',
              _dark: { color: 'base.400' },
            }}
          ></Icon>
          <IAINodeResizer />
        </Flex>
      </NodeWrapper>
    );
  }

  return (
    <NodeWrapper selected={selected}>
      <IAINodeHeader
        nodeId={nodeId}
        title={template.title}
        description={template.description}
      />
      <Flex
        sx={{
          flexDirection: 'column',
          borderBottomRadius: 'md',
          py: 2,
          bg: 'base.200',
          _dark: { bg: 'base.800' },
        }}
      >
        <IAINodeOutputs nodeId={nodeId} outputs={outputs} template={template} />
        <IAINodeInputs nodeId={nodeId} inputs={inputs} template={template} />
      </Flex>
      <IAINodeResizer />
    </NodeWrapper>
  );
});

InvocationComponent.displayName = 'InvocationComponent';

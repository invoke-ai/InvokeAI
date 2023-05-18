import { NodeProps } from 'reactflow';
import { Box, Flex, Icon, useToken } from '@chakra-ui/react';
import { FaExclamationCircle } from 'react-icons/fa';
import { InvocationTemplate, InvocationValue } from '../types/types';

import { memo, PropsWithChildren, useMemo } from 'react';
import IAINodeOutputs from './IAINode/IAINodeOutputs';
import IAINodeInputs from './IAINode/IAINodeInputs';
import IAINodeHeader from './IAINode/IAINodeHeader';
import IAINodeResizer from './IAINode/IAINodeResizer';
import { RootState } from 'app/store/store';
import { AnyInvocationType } from 'services/events/types';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { NODE_MIN_WIDTH } from 'app/constants';

type InvocationComponentWrapperProps = PropsWithChildren & {
  selected: boolean;
};

const InvocationComponentWrapper = (props: InvocationComponentWrapperProps) => {
  const [nodeSelectedOutline, nodeShadow] = useToken('shadows', [
    'nodeSelectedOutline',
    'dark-lg',
  ]);

  return (
    <Box
      sx={{
        position: 'relative',
        borderRadius: 'md',
        minWidth: NODE_MIN_WIDTH,
        boxShadow: props.selected
          ? `${nodeSelectedOutline}, ${nodeShadow}`
          : `${nodeShadow}`,
      }}
    >
      {props.children}
    </Box>
  );
};

const makeTemplateSelector = (type: AnyInvocationType) =>
  createSelector(
    [(state: RootState) => state.nodes],
    (nodes) => {
      const template = nodes.invocationTemplates[type];
      if (!template) {
        return;
      }
      return template;
    },
    {
      memoizeOptions: {
        resultEqualityCheck: (
          a: InvocationTemplate | undefined,
          b: InvocationTemplate | undefined
        ) => a !== undefined && b !== undefined && a.type === b.type,
      },
    }
  );

export const InvocationComponent = memo((props: NodeProps<InvocationValue>) => {
  const { id: nodeId, data, selected } = props;
  const { type, inputs, outputs } = data;

  const templateSelector = useMemo(() => makeTemplateSelector(type), [type]);

  const template = useAppSelector(templateSelector);

  if (!template) {
    return (
      <InvocationComponentWrapper selected={selected}>
        <Flex sx={{ alignItems: 'center', justifyContent: 'center' }}>
          <Icon color="base.400" boxSize={32} as={FaExclamationCircle}></Icon>
          <IAINodeResizer />
        </Flex>
      </InvocationComponentWrapper>
    );
  }

  return (
    <InvocationComponentWrapper selected={selected}>
      <IAINodeHeader nodeId={nodeId} template={template} />
      <Flex
        sx={{
          flexDirection: 'column',
          borderBottomRadius: 'md',
          bg: 'base.800',
          py: 2,
        }}
      >
        <IAINodeOutputs nodeId={nodeId} outputs={outputs} template={template} />
        <IAINodeInputs nodeId={nodeId} inputs={inputs} template={template} />
      </Flex>
      <IAINodeResizer />
    </InvocationComponentWrapper>
  );
});

InvocationComponent.displayName = 'InvocationComponent';

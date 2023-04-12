import { NodeProps, useReactFlow } from 'reactflow';
import {
  Box,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  HStack,
  Tooltip,
  Icon,
  Code,
  Text,
} from '@chakra-ui/react';
import { FaInfoCircle } from 'react-icons/fa';
import { Invocation } from '../types';
import { InputFieldComponent } from './InputFieldComponent';
import { FieldHandle } from './FieldHandle';
import { isEqual, map, size } from 'lodash';
import { memo, useMemo } from 'react';
import { useIsValidConnection } from '../hooks/useIsValidConnection';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';

const connectedInputFieldsSelector = createSelector(
  (state: RootState) => state.nodes.edges,
  (edges) => {
    return edges.map((e) => e.targetHandle);
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const InvocationComponent = memo((props: NodeProps<Invocation>) => {
  const { id, data, selected } = props;
  const { type, title, description, inputs, outputs } = data;

  const isValidConnection = useIsValidConnection();

  const connectedInputs = useAppSelector(connectedInputFieldsSelector);
  // TODO: determine if a field/handle is connected and disable the input if so

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
      <Flex flexDirection="column" gap={2}>
        <>
          <Code>{id}</Code>
          <HStack justifyContent="space-between">
            <Heading size="sm" fontWeight={500} color="base.100">
              {title}
            </Heading>
            <Tooltip
              label={description}
              placement="top"
              hasArrow
              shouldWrapChildren
            >
              <Icon color="base.300" as={FaInfoCircle} />
            </Tooltip>
          </HStack>
          {map(inputs, (input, i) => {
            const isConnected = connectedInputs.includes(input.name);
            return (
              <Box
                key={i}
                position="relative"
                p={2}
                borderWidth={1}
                borderRadius="md"
                sx={{
                  borderColor:
                    !isConnected && input.connectionType === 'always'
                      ? 'warning.400'
                      : undefined,
                }}
              >
                <FormControl isDisabled={isConnected}>
                  <HStack justifyContent="space-between" alignItems="center">
                    <FormLabel>{input.title}</FormLabel>
                    <Tooltip
                      label={input.description}
                      placement="top"
                      hasArrow
                      shouldWrapChildren
                    >
                      <Icon color="base.400" as={FaInfoCircle} />
                    </Tooltip>
                  </HStack>
                  <InputFieldComponent nodeId={id} field={input} />
                </FormControl>
                {input.connectionType !== 'never' && (
                  <FieldHandle
                    nodeId={id}
                    field={input}
                    isValidConnection={isValidConnection}
                    handleType="target"
                  />
                )}
              </Box>
            );
          })}
          {map(outputs).map((output, i) => {
            // const top = `${(100 / (size(outputs) + 1)) * (i + 1)}%`;
            const { name, title } = output;
            return (
              <Box
                key={name}
                position="relative"
                p={2}
                borderWidth={1}
                borderRadius="md"
              >
                <FormControl>
                  <FormLabel textAlign="end">{title} Output</FormLabel>
                </FormControl>
                <FieldHandle
                  key={name}
                  nodeId={id}
                  field={output}
                  isValidConnection={isValidConnection}
                  handleType="source"
                />
              </Box>
            );
          })}
        </>
      </Flex>
      <Flex></Flex>
    </Box>
  );
});

InvocationComponent.displayName = 'InvocationComponent';

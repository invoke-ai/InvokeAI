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
import { FaExclamationCircle, FaInfoCircle } from 'react-icons/fa';
import { InvocationValue } from '../types/types';
import { InputFieldComponent } from './InputFieldComponent';
import { FieldHandle } from './FieldHandle';
import { isEqual, map, size } from 'lodash';
import { memo, useMemo, useRef } from 'react';
import { useIsValidConnection } from '../hooks/useIsValidConnection';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { useGetInvocationTemplate } from '../hooks/useInvocationTemplate';

const connectedInputFieldsSelector = createSelector(
  [(state: RootState) => state.nodes.edges],
  (edges) => {
    // return edges.map((e) => e.targetHandle);
    return edges;
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const InvocationComponent = memo((props: NodeProps<InvocationValue>) => {
  const { id: nodeId, data, selected } = props;
  const { type, inputs, outputs } = data;

  const isValidConnection = useIsValidConnection();

  const connectedInputs = useAppSelector(connectedInputFieldsSelector);
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
        </Flex>
      </Box>
    );
  }

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
          <Code>{nodeId}</Code>
          <HStack justifyContent="space-between">
            <Heading size="sm" fontWeight={500} color="base.100">
              {template.current.title}
            </Heading>
            <Tooltip
              label={template.current.description}
              placement="top"
              hasArrow
              shouldWrapChildren
            >
              <Icon color="base.300" as={FaInfoCircle} />
            </Tooltip>
          </HStack>
          {map(inputs, (input, i) => {
            const { id: fieldId } = input;
            const inputTemplate = template.current?.inputs[input.name];

            if (!inputTemplate) {
              return (
                <Box
                  key={fieldId}
                  position="relative"
                  p={2}
                  borderWidth={1}
                  borderRadius="md"
                  sx={{
                    borderColor: 'error.400',
                  }}
                >
                  <FormControl isDisabled={true}>
                    <HStack justifyContent="space-between" alignItems="center">
                      <FormLabel>Unknown input: {input.name}</FormLabel>
                    </HStack>
                  </FormControl>
                </Box>
              );
            }

            const isConnected = Boolean(
              connectedInputs.filter((connectedInput) => {
                return (
                  connectedInput.target === nodeId &&
                  connectedInput.targetHandle === input.name
                );
              }).length
            );

            return (
              <Box
                key={fieldId}
                position="relative"
                p={2}
                borderWidth={1}
                borderRadius="md"
                sx={{
                  borderColor:
                    !isConnected &&
                    ['always', 'connectionOnly'].includes(
                      String(inputTemplate?.inputRequirement)
                    ) &&
                    input.value === undefined
                      ? 'warning.400'
                      : undefined,
                }}
              >
                <FormControl isDisabled={isConnected}>
                  <HStack justifyContent="space-between" alignItems="center">
                    <FormLabel>{inputTemplate?.title}</FormLabel>
                    <Tooltip
                      label={inputTemplate?.description}
                      placement="top"
                      hasArrow
                      shouldWrapChildren
                    >
                      <Icon color="base.400" as={FaInfoCircle} />
                    </Tooltip>
                  </HStack>
                  <InputFieldComponent
                    nodeId={nodeId}
                    field={input}
                    template={inputTemplate}
                  />
                </FormControl>
                {!['never', 'directOnly'].includes(
                  inputTemplate?.inputRequirement ?? ''
                ) && (
                  <FieldHandle
                    nodeId={nodeId}
                    field={inputTemplate}
                    isValidConnection={isValidConnection}
                    handleType="target"
                  />
                )}
              </Box>
            );
          })}
          {map(outputs).map((output, i) => {
            const outputTemplate = template.current?.outputs[output.name];

            const isConnected = Boolean(
              connectedInputs.filter((connectedInput) => {
                return (
                  connectedInput.source === nodeId &&
                  connectedInput.sourceHandle === output.name
                );
              }).length
            );

            if (!outputTemplate) {
              return (
                <Box
                  key={output.id}
                  position="relative"
                  p={2}
                  borderWidth={1}
                  borderRadius="md"
                  sx={{
                    borderColor: 'error.400',
                  }}
                >
                  <FormControl isDisabled={true}>
                    <HStack justifyContent="space-between" alignItems="center">
                      <FormLabel>Unknown output: {output.name}</FormLabel>
                    </HStack>
                  </FormControl>
                </Box>
              );
            }

            return (
              <Box
                key={output.id}
                position="relative"
                p={2}
                borderWidth={1}
                borderRadius="md"
              >
                <FormControl isDisabled={isConnected}>
                  <FormLabel textAlign="end">
                    {outputTemplate?.title} Output
                  </FormLabel>
                </FormControl>
                <FieldHandle
                  key={output.id}
                  nodeId={nodeId}
                  field={outputTemplate}
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

import { Connection, NodeProps, useReactFlow } from 'reactflow';
import {
  Box,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  HStack,
  Tooltip,
  Icon,
} from '@chakra-ui/react';
import { FaInfoCircle } from 'react-icons/fa';
import { Invocation } from '../types';
import { InputFieldComponent } from './InputFieldComponent';
import { FieldHandle } from './FieldHandle';
import { map, size } from 'lodash';
import { memo, useCallback } from 'react';

export const InvocationComponent = memo((props: NodeProps<Invocation>) => {
  const { id, data, selected } = props;
  const { type, title, description, inputs, outputs } = data;

  const flow = useReactFlow();

  // Check if an in-progress connection is valid
  const isValidConnection = useCallback(
    (connection: Connection): boolean => {
      const edges = flow.getEdges();

      // Connection is invalid if target already has a connection
      if (
        edges.find((edge) => {
          return (
            edge.target === connection.target &&
            edge.targetHandle === connection.targetHandle
          );
        })
      ) {
        return false;
      }

      // Find the source and target nodes...
      if (connection.source && connection.target) {
        const sourceNode = flow.getNode(connection.source);
        const targetNode = flow.getNode(connection.target);

        // Conditional guards against undefined nodes/handles
        if (
          sourceNode &&
          targetNode &&
          connection.sourceHandle &&
          connection.targetHandle
        ) {
          // connection types must be the same for a connection
          return (
            sourceNode.data.outputs[connection.sourceHandle].type ===
            targetNode.data.inputs[connection.targetHandle].type
          );
        }
      }

      // Default to invalid
      return false;
    },
    [flow]
  );

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
            return (
              <Box
                key={i}
                position="relative"
                p={2}
                borderWidth={1}
                borderRadius="md"
              >
                <FormControl>
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
                <FieldHandle
                  nodeId={id}
                  field={input}
                  isValidConnection={isValidConnection}
                  handleType="target"
                />
              </Box>
            );
          })}
        </>
      </Flex>
      <Flex>
        {map(outputs).map((output, i) => {
          const top = `${(100 / (size(outputs) + 1)) * (i + 1)}%`;

          return (
            <FieldHandle
              key={i}
              nodeId={id}
              field={output}
              isValidConnection={isValidConnection}
              handleType="source"
              styles={{ top }}
            />
          );
        })}
      </Flex>
    </Box>
  );
});

InvocationComponent.displayName = 'InvocationComponent';

import { NodeProps } from 'reactflow';
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
import { buildFieldComponent } from '../util/buildFieldComponent';
import {
  buildInputHandleComponent,
  buildOutputHandleComponent,
} from '../util/buildHandleComponent';

export const InvocationComponent = (props: NodeProps<Invocation>) => {
  const { id, data, selected } = props;
  const { type, title, description, inputs, outputs } = data;

  return (
    <Box
      sx={{
        padding: 4,
        bg: 'base.800',
        borderRadius: 'md',
        boxShadow: 'dark-lg',
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
          {inputs.map((input, i) => {
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
                  {buildFieldComponent(id, input)}
                </FormControl>
                {buildInputHandleComponent(id, input)}
              </Box>
            );
          })}
        </>
      </Flex>
      {outputs.map((output, i) => {
        const top = `${(100 / (outputs.length + 1)) * (i + 1)}%`;
        return buildOutputHandleComponent(id, output, top);
      })}
    </Box>
  );
};

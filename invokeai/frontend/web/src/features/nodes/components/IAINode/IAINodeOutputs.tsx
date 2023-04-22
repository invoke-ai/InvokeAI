import {
  InvocationTemplate,
  OutputFieldTemplate,
  OutputFieldValue,
} from 'features/nodes/types/types';
import { MutableRefObject, ReactNode } from 'react';
import { map } from 'lodash';
import { useAppSelector } from 'app/storeHooks';
import { RootState } from 'app/store';
import { Box, Flex, FormControl, FormLabel, HStack } from '@chakra-ui/react';
import { FieldHandle } from '../FieldHandle';
import { useIsValidConnection } from 'features/nodes/hooks/useIsValidConnection';

interface IAINodeOutputProps {
  nodeId: string;
  output: OutputFieldValue;
  template?: OutputFieldTemplate | undefined;
  connected: boolean;
}

function IAINodeOutput(props: IAINodeOutputProps) {
  const { nodeId, output, template, connected } = props;
  const isValidConnection = useIsValidConnection();

  return (
    <Box key={output.id} position="relative">
      <FormControl isDisabled={!template ? true : connected} paddingRight={3}>
        {!template ? (
          <HStack justifyContent="space-between" alignItems="center">
            <FormLabel color="error.400">
              Unknown Output: {output.name}
            </FormLabel>
          </HStack>
        ) : (
          <>
            <FormLabel textAlign="end" padding={1}>
              {template?.title}
            </FormLabel>
            <FieldHandle
              key={output.id}
              nodeId={nodeId}
              field={template}
              isValidConnection={isValidConnection}
              handleType="source"
            />
          </>
        )}
      </FormControl>
    </Box>
  );
}

interface IAINodeOutputsProps {
  nodeId: string;
  template: MutableRefObject<InvocationTemplate | undefined>;
  outputs: Record<string, OutputFieldValue>;
}

export default function IAINodeOutputs(props: IAINodeOutputsProps) {
  const { nodeId, template, outputs } = props;

  const connectedInputs = useAppSelector(
    (state: RootState) => state.nodes.edges
  );

  const renderIAINodeOutputs = () => {
    const IAINodeOutputsToRender: ReactNode[] = [];
    const outputSockets = map(outputs);

    outputSockets.forEach((outputSocket) => {
      const outputTemplate = template.current?.outputs[outputSocket.name];

      const isConnected = Boolean(
        connectedInputs.filter((connectedInput) => {
          return (
            connectedInput.source === nodeId &&
            connectedInput.sourceHandle === outputSocket.name
          );
        }).length
      );

      IAINodeOutputsToRender.push(
        <IAINodeOutput
          nodeId={nodeId}
          output={outputSocket}
          template={outputTemplate}
          connected={isConnected}
        />
      );
    });

    return <Flex flexDir="column">{IAINodeOutputsToRender}</Flex>;
  };

  return renderIAINodeOutputs();
}

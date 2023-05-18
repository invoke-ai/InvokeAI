import {
  InvocationTemplate,
  OutputFieldTemplate,
  OutputFieldValue,
} from 'features/nodes/types/types';
import { memo, ReactNode, useCallback } from 'react';
import { map } from 'lodash-es';
import { useAppSelector } from 'app/store/storeHooks';
import { RootState } from 'app/store/store';
import { Box, Flex, FormControl, FormLabel, HStack } from '@chakra-ui/react';
import FieldHandle from '../FieldHandle';
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
    <Box position="relative">
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
  template: InvocationTemplate;
  outputs: Record<string, OutputFieldValue>;
}

const IAINodeOutputs = (props: IAINodeOutputsProps) => {
  const { nodeId, template, outputs } = props;

  const edges = useAppSelector((state: RootState) => state.nodes.edges);

  const renderIAINodeOutputs = useCallback(() => {
    const IAINodeOutputsToRender: ReactNode[] = [];
    const outputSockets = map(outputs);

    outputSockets.forEach((outputSocket) => {
      const outputTemplate = template.outputs[outputSocket.name];

      const isConnected = Boolean(
        edges.filter((connectedInput) => {
          return (
            connectedInput.source === nodeId &&
            connectedInput.sourceHandle === outputSocket.name
          );
        }).length
      );

      IAINodeOutputsToRender.push(
        <IAINodeOutput
          key={outputSocket.id}
          nodeId={nodeId}
          output={outputSocket}
          template={outputTemplate}
          connected={isConnected}
        />
      );
    });

    return <Flex flexDir="column">{IAINodeOutputsToRender}</Flex>;
  }, [edges, nodeId, outputs, template.outputs]);

  return renderIAINodeOutputs();
};

export default memo(IAINodeOutputs);

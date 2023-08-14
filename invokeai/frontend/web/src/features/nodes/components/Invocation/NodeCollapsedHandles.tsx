import { useColorModeValue } from '@chakra-ui/react';
import { useChakraThemeTokens } from 'common/hooks/useChakraThemeTokens';
import {
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { map } from 'lodash-es';
import { CSSProperties, memo, useMemo } from 'react';
import { Handle, NodeProps, Position } from 'reactflow';

interface Props {
  nodeProps: NodeProps<InvocationNodeData>;
  nodeTemplate: InvocationTemplate;
}

const NodeCollapsedHandles = (props: Props) => {
  const { data } = props.nodeProps;
  const { base400, base600 } = useChakraThemeTokens();
  const backgroundColor = useColorModeValue(base400, base600);

  const dummyHandleStyles: CSSProperties = useMemo(
    () => ({
      borderWidth: 0,
      borderRadius: '3px',
      width: '1rem',
      height: '1rem',
      backgroundColor,
      zIndex: -1,
    }),
    [backgroundColor]
  );

  return (
    <>
      <Handle
        type="target"
        id={`${data.id}-collapsed-target`}
        isConnectable={false}
        position={Position.Left}
        style={{ ...dummyHandleStyles, left: '-0.5rem' }}
      />
      {map(data.inputs, (input) => (
        <Handle
          key={`${data.id}-${input.name}-collapsed-input-handle`}
          type="target"
          id={input.name}
          isValidConnection={() => false}
          position={Position.Left}
          style={{ visibility: 'hidden' }}
        />
      ))}
      <Handle
        type="source"
        id={`${data.id}-collapsed-source`}
        isValidConnection={() => false}
        isConnectable={false}
        position={Position.Right}
        style={{ ...dummyHandleStyles, right: '-0.5rem' }}
      />
      {map(data.outputs, (output) => (
        <Handle
          key={`${data.id}-${output.name}-collapsed-output-handle`}
          type="source"
          id={output.name}
          isValidConnection={() => false}
          position={Position.Right}
          style={{ visibility: 'hidden' }}
        />
      ))}
    </>
  );
};

export default memo(NodeCollapsedHandles);

import { useChakraThemeTokens } from 'common/hooks/useChakraThemeTokens';
import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { isInvocationNodeData } from 'features/nodes/types/invocation';
import { map } from 'lodash-es';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import { Handle, Position } from 'reactflow';

interface Props {
  nodeId: string;
}

const InvocationNodeCollapsedHandles = ({ nodeId }: Props) => {
  const data = useNodeData(nodeId);
  const { base600 } = useChakraThemeTokens();

  const dummyHandleStyles: CSSProperties = useMemo(
    () => ({
      borderWidth: 0,
      borderRadius: '3px',
      width: '1rem',
      height: '1rem',
      backgroundColor: base600,
      zIndex: -1,
    }),
    [base600]
  );

  if (!isInvocationNodeData(data)) {
    return null;
  }

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
          isConnectable={false}
          position={Position.Left}
          style={{ visibility: 'hidden' }}
        />
      ))}
      <Handle
        type="source"
        id={`${data.id}-collapsed-source`}
        isConnectable={false}
        position={Position.Right}
        style={{ ...dummyHandleStyles, right: '-0.5rem' }}
      />
      {map(data.outputs, (output) => (
        <Handle
          key={`${data.id}-${output.name}-collapsed-output-handle`}
          type="source"
          id={output.name}
          isConnectable={false}
          position={Position.Right}
          style={{ visibility: 'hidden' }}
        />
      ))}
    </>
  );
};

export default memo(InvocationNodeCollapsedHandles);

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

const hiddenHandleStyles: CSSProperties = { visibility: 'hidden' };

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

  const collapsedTargetStyles: CSSProperties = useMemo(
    () => ({ ...dummyHandleStyles, left: '-0.5rem' }),
    [dummyHandleStyles]
  );
  const collapsedSourceStyles: CSSProperties = useMemo(
    () => ({ ...dummyHandleStyles, right: '-0.5rem' }),
    [dummyHandleStyles]
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
        style={collapsedTargetStyles}
      />
      {map(data.inputs, (input) => (
        <Handle
          key={`${data.id}-${input.name}-collapsed-input-handle`}
          type="target"
          id={input.name}
          isConnectable={false}
          position={Position.Left}
          style={hiddenHandleStyles}
        />
      ))}
      <Handle
        type="source"
        id={`${data.id}-collapsed-source`}
        isConnectable={false}
        position={Position.Right}
        style={collapsedSourceStyles}
      />
      {map(data.outputs, (output) => (
        <Handle
          key={`${data.id}-${output.name}-collapsed-output-handle`}
          type="source"
          id={output.name}
          isConnectable={false}
          position={Position.Right}
          style={hiddenHandleStyles}
        />
      ))}
    </>
  );
};

export default memo(InvocationNodeCollapsedHandles);

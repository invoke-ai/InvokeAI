import { useChakraThemeTokens } from 'common/hooks/useChakraThemeTokens';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { map } from 'lodash-es';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import { Handle, Position } from 'reactflow';

interface Props {
  nodeId: string;
}

const hiddenHandleStyles: CSSProperties = { visibility: 'hidden' };

const InvocationNodeCollapsedHandles = ({ nodeId }: Props) => {
  const template = useNodeTemplate(nodeId);
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

  if (!template) {
    return null;
  }

  return (
    <>
      <Handle
        type="target"
        id={`${nodeId}-collapsed-target`}
        isConnectable={false}
        position={Position.Left}
        style={collapsedTargetStyles}
      />
      {map(template.inputs, (input) => (
        <Handle
          key={`${nodeId}-${input.name}-collapsed-input-handle`}
          type="target"
          id={input.name}
          isConnectable={false}
          position={Position.Left}
          style={hiddenHandleStyles}
        />
      ))}
      <Handle
        type="source"
        id={`${nodeId}-collapsed-source`}
        isConnectable={false}
        position={Position.Right}
        style={collapsedSourceStyles}
      />
      {map(template.outputs, (output) => (
        <Handle
          key={`${nodeId}-${output.name}-collapsed-output-handle`}
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

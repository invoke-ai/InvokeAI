import { Handle, Position } from '@xyflow/react';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { map } from 'lodash-es';
import type { CSSProperties } from 'react';
import { memo } from 'react';

interface Props {
  nodeId: string;
}

const hiddenHandleStyles: CSSProperties = { visibility: 'hidden' };
const collapsedHandleStyles: CSSProperties = {
  borderWidth: 0,
  borderRadius: '3px',
  width: '1rem',
  height: '1rem',
  backgroundColor: 'var(--invoke-colors-base-600)',
  zIndex: -1,
};

const InvocationNodeCollapsedHandles = ({ nodeId }: Props) => {
  const template = useNodeTemplate(nodeId);

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
        style={collapsedHandleStyles}
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
        style={collapsedHandleStyles}
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

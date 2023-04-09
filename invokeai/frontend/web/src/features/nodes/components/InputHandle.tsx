import { Tooltip } from '@chakra-ui/react';
import { Handle, Position } from 'reactflow';
import { FIELDS, InputField } from '../types';

type InputHandleProps = {
  nodeId: string;
  field: InputField;
};

export const InputHandle = (props: InputHandleProps) => {
  const { nodeId, field } = props;
  const { title, type, description } = field;
  return (
    <Tooltip
      key={type}
      label={`${title}: ${description}`}
      placement="start"
      hasArrow
    >
      <Handle
        type="target"
        id={type}
        position={Position.Left}
        style={{
          position: 'absolute',
          left: '-1.5rem',
          width: '1rem',
          height: '1rem',
          backgroundColor: `var(--invokeai-colors-${FIELDS[type].color}-500)`,
        }}
      />
    </Tooltip>
  );
};

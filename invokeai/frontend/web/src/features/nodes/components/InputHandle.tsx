import { Tooltip } from '@chakra-ui/react';
import { Handle, Position, Connection } from 'reactflow';
import { FIELDS, HANDLE_TOOLTIP_OPEN_DELAY } from '../constants';
import { InputField } from '../types';

type InputHandleProps = {
  nodeId: string;
  field: InputField;
  isValidConnection: (connection: Connection) => boolean;
};

export const InputHandle = (props: InputHandleProps) => {
  const { nodeId, field, isValidConnection } = props;
  const { name, title, type, description } = field;
  return (
    <Tooltip
      key={name}
      label={`${title} (${type})`}
      placement="start"
      hasArrow
      openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
    >
      <Handle
        type="target"
        id={name}
        isValidConnection={isValidConnection}
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

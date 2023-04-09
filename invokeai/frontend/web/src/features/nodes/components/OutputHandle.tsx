import { Tooltip } from '@chakra-ui/react';
import { Handle, Position, Connection } from 'reactflow';
import { FIELDS, HANDLE_TOOLTIP_OPEN_DELAY } from '../constants';
import { OutputField } from '../types';

type OutputHandleProps = {
  nodeId: string;
  field: OutputField;
  isValidConnection: (connection: Connection) => boolean;
  top: string;
};

export const OutputHandle = (props: OutputHandleProps) => {
  const { nodeId, field, isValidConnection, top } = props;
  const { name, title, type, description } = field;

  return (
    <Tooltip
      label={`${title} (${type})`}
      placement="end"
      hasArrow
      openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
    >
      <Handle
        type="source"
        id={name}
        isValidConnection={isValidConnection}
        position={Position.Right}
        style={{
          position: 'absolute',
          top,
          right: '-0.5rem',
          width: '1rem',
          height: '1rem',
          backgroundColor: `var(--invokeai-colors-${FIELDS[type].color}-500)`,
        }}
      />
    </Tooltip>
  );
};

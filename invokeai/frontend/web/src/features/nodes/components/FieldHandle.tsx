import { Tooltip } from '@chakra-ui/react';
import { CSSProperties } from 'react';
import { Handle, Position, Connection, HandleType } from 'reactflow';
import { FIELDS, HANDLE_TOOLTIP_OPEN_DELAY } from '../constants';
// import { useConnectionEventStyles } from '../hooks/useConnectionEventStyles';
import { InputField, OutputField } from '../types';

const handleBaseStyles: CSSProperties = {
  position: 'absolute',
  width: '1rem',
  height: '1rem',
  borderWidth: 0,
};

const inputHandleStyles: CSSProperties = {
  left: '-1.7rem',
};

const outputHandleStyles: CSSProperties = {
  right: '-0.5rem',
};

type FieldHandleProps = {
  nodeId: string;
  field: InputField | OutputField;
  isValidConnection: (connection: Connection) => boolean;
  handleType: HandleType;
  styles?: CSSProperties;
};

export const FieldHandle = (props: FieldHandleProps) => {
  const { nodeId, field, isValidConnection, handleType, styles } = props;
  const { name, title, type, description } = field;

  // this needs to iterate over every candicate target node, calculating graph cycles
  // WIP
  // const connectionEventStyles = useConnectionEventStyles(
  //   nodeId,
  //   type,
  //   handleType
  // );

  return (
    <Tooltip
      key={name}
      label={`${title} (${type})`}
      placement={handleType === 'target' ? 'start' : 'end'}
      hasArrow
      openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
    >
      <Handle
        type={handleType}
        id={name}
        isValidConnection={isValidConnection}
        position={handleType === 'target' ? Position.Left : Position.Right}
        style={{
          backgroundColor: `var(--invokeai-colors-${FIELDS[type].color}-500)`,
          ...styles,
          ...(handleType === 'target' ? inputHandleStyles : outputHandleStyles),
          ...handleBaseStyles,
          // ...connectionEventStyles,
        }}
      />
    </Tooltip>
  );
};

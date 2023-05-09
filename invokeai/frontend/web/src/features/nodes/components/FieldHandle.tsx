import { Tooltip } from '@chakra-ui/react';
import { CSSProperties, memo } from 'react';
import { Handle, Position, Connection, HandleType } from 'reactflow';
import { FIELDS, HANDLE_TOOLTIP_OPEN_DELAY } from '../types/constants';
// import { useConnectionEventStyles } from '../hooks/useConnectionEventStyles';
import { InputFieldTemplate, OutputFieldTemplate } from '../types/types';

const handleBaseStyles: CSSProperties = {
  position: 'absolute',
  width: '1rem',
  height: '1rem',
  borderWidth: 0,
};

const inputHandleStyles: CSSProperties = {
  left: '-1rem',
};

const outputHandleStyles: CSSProperties = {
  right: '-0.5rem',
};

// const requiredConnectionStyles: CSSProperties = {
//   boxShadow: '0 0 0.5rem 0.5rem var(--invokeai-colors-error-400)',
// };

type FieldHandleProps = {
  nodeId: string;
  field: InputFieldTemplate | OutputFieldTemplate;
  isValidConnection: (connection: Connection) => boolean;
  handleType: HandleType;
  styles?: CSSProperties;
};

const FieldHandle = (props: FieldHandleProps) => {
  const { field, isValidConnection, handleType, styles } = props;
  const { name, type } = field;

  return (
    <Tooltip
      label={type}
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
          backgroundColor: FIELDS[type].colorCssVar,
          ...styles,
          ...handleBaseStyles,
          ...(handleType === 'target' ? inputHandleStyles : outputHandleStyles),
          // ...(inputRequirement === 'always' ? requiredConnectionStyles : {}),
          // ...connectionEventStyles,
        }}
      />
    </Tooltip>
  );
};

export default memo(FieldHandle);

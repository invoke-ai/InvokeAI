import { Tooltip } from '@chakra-ui/react';
import { CSSProperties, useMemo } from 'react';
import {
  Handle,
  Position,
  Connection,
  HandleType,
  useReactFlow,
} from 'reactflow';
import { FIELDS, HANDLE_TOOLTIP_OPEN_DELAY } from '../constants';
// import { useConnectionEventStyles } from '../hooks/useConnectionEventStyles';
import { InputFieldTemplate, OutputFieldTemplate } from '../types';

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
  right: '-1.7rem',
};

const requiredConnectionStyles: CSSProperties = {
  boxShadow: '0 0 0.5rem 0.5rem var(--invokeai-colors-error-400)',
};

type FieldHandleProps = {
  nodeId: string;
  field: InputFieldTemplate | OutputFieldTemplate;
  isValidConnection: (connection: Connection) => boolean;
  handleType: HandleType;
  styles?: CSSProperties;
};

export const FieldHandle = (props: FieldHandleProps) => {
  const { nodeId, field, isValidConnection, handleType, styles } = props;
  const { name, title, type, description } = field;

  return (
    <Tooltip
      key={name}
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

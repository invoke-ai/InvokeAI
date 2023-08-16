import { Tooltip } from '@chakra-ui/react';
import { CSSProperties, memo, useMemo } from 'react';
import { Handle, HandleType, NodeProps, Position } from 'reactflow';
import {
  FIELDS,
  HANDLE_TOOLTIP_OPEN_DELAY,
  colorTokenToCssVar,
} from '../../types/constants';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
  OutputFieldTemplate,
  OutputFieldValue,
} from '../../types/types';

export const handleBaseStyles: CSSProperties = {
  position: 'absolute',
  width: '1rem',
  height: '1rem',
  borderWidth: 0,
  zIndex: 1,
};

export const inputHandleStyles: CSSProperties = {
  left: '-1rem',
};

export const outputHandleStyles: CSSProperties = {
  right: '-0.5rem',
};

type FieldHandleProps = {
  nodeProps: NodeProps<InvocationNodeData>;
  nodeTemplate: InvocationTemplate;
  field: InputFieldValue | OutputFieldValue;
  fieldTemplate: InputFieldTemplate | OutputFieldTemplate;
  handleType: HandleType;
  isConnectionInProgress: boolean;
  isConnectionStartField: boolean;
  connectionError: string | null;
};

const FieldHandle = (props: FieldHandleProps) => {
  const {
    fieldTemplate,
    handleType,
    isConnectionInProgress,
    isConnectionStartField,
    connectionError,
  } = props;
  const { name, type } = fieldTemplate;
  const { color, title } = FIELDS[type];

  const styles: CSSProperties = useMemo(() => {
    const s: CSSProperties = {
      backgroundColor: colorTokenToCssVar(color),
      position: 'absolute',
      width: '1rem',
      height: '1rem',
      borderWidth: 0,
      zIndex: 1,
    };

    if (handleType === 'target') {
      s.insetInlineStart = '-1rem';
    } else {
      s.insetInlineEnd = '-1rem';
    }

    if (isConnectionInProgress && !isConnectionStartField && connectionError) {
      s.filter = 'opacity(0.4) grayscale(0.7)';
    }

    if (isConnectionInProgress && connectionError) {
      if (isConnectionStartField) {
        s.cursor = 'grab';
      } else {
        s.cursor = 'not-allowed';
      }
    } else {
      s.cursor = 'crosshair';
    }

    return s;
  }, [
    color,
    connectionError,
    handleType,
    isConnectionInProgress,
    isConnectionStartField,
  ]);

  const tooltip = useMemo(() => {
    if (isConnectionInProgress && isConnectionStartField) {
      return title;
    }
    if (isConnectionInProgress && connectionError) {
      return connectionError ?? title;
    }
    return title;
  }, [connectionError, isConnectionInProgress, isConnectionStartField, title]);

  return (
    <Tooltip
      label={tooltip}
      placement={handleType === 'target' ? 'start' : 'end'}
      hasArrow
      openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
    >
      <Handle
        type={handleType}
        id={name}
        position={handleType === 'target' ? Position.Left : Position.Right}
        style={styles}
      />
    </Tooltip>
  );
};

export default memo(FieldHandle);

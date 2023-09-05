import { Tooltip } from '@chakra-ui/react';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import {
  COLLECTION_TYPES,
  FIELDS,
  HANDLE_TOOLTIP_OPEN_DELAY,
  MODEL_TYPES,
  POLYMORPHIC_TYPES,
} from 'features/nodes/types/constants';
import {
  InputFieldTemplate,
  OutputFieldTemplate,
} from 'features/nodes/types/types';
import { CSSProperties, memo, useMemo } from 'react';
import { Handle, HandleType, Position } from 'reactflow';

export const handleBaseStyles: CSSProperties = {
  position: 'absolute',
  width: '1rem',
  height: '1rem',
  borderWidth: 0,
  zIndex: 1,
};
``;

export const inputHandleStyles: CSSProperties = {
  left: '-1rem',
};

export const outputHandleStyles: CSSProperties = {
  right: '-0.5rem',
};

type FieldHandleProps = {
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
  const { color: typeColor, title } = FIELDS[type];

  const styles: CSSProperties = useMemo(() => {
    const isCollectionType = COLLECTION_TYPES.includes(type);
    const isPolymorphicType = POLYMORPHIC_TYPES.includes(type);
    const isModelType = MODEL_TYPES.includes(type);
    const color = colorTokenToCssVar(typeColor);
    const s: CSSProperties = {
      backgroundColor:
        isCollectionType || isPolymorphicType
          ? 'var(--invokeai-colors-base-900)'
          : color,
      position: 'absolute',
      width: '1rem',
      height: '1rem',
      borderWidth: isCollectionType || isPolymorphicType ? 4 : 0,
      borderStyle: 'solid',
      borderColor: color,
      borderRadius: isModelType ? 4 : '100%',
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
    connectionError,
    handleType,
    isConnectionInProgress,
    isConnectionStartField,
    type,
    typeColor,
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

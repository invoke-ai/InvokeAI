import { Tooltip } from '@invoke-ai/ui-library';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { getFieldColor } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import { useFieldTypeName } from 'features/nodes/hooks/usePrettyFieldType';
import { HANDLE_TOOLTIP_OPEN_DELAY, MODEL_TYPES } from 'features/nodes/types/constants';
import type { FieldInputTemplate, FieldOutputTemplate } from 'features/nodes/types/field';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import type { HandleType } from 'reactflow';
import { Handle, Position } from 'reactflow';

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
  fieldTemplate: FieldInputTemplate | FieldOutputTemplate;
  handleType: HandleType;
  isConnectionInProgress: boolean;
  isConnectionStartField: boolean;
  connectionError?: string;
};

const FieldHandle = (props: FieldHandleProps) => {
  const { fieldTemplate, handleType, isConnectionInProgress, isConnectionStartField, connectionError } = props;
  const { name } = fieldTemplate;
  const type = fieldTemplate.type;
  const fieldTypeName = useFieldTypeName(type);
  const styles: CSSProperties = useMemo(() => {
    const isModelType = MODEL_TYPES.some((t) => t === type.name);
    const color = getFieldColor(type);
    const s: CSSProperties = {
      backgroundColor: type.isCollection || type.isCollectionOrScalar ? colorTokenToCssVar('base.900') : color,
      position: 'absolute',
      width: '1rem',
      height: '1rem',
      borderWidth: type.isCollection || type.isCollectionOrScalar ? 4 : 0,
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
  }, [connectionError, handleType, isConnectionInProgress, isConnectionStartField, type]);

  const tooltip = useMemo(() => {
    if (isConnectionInProgress && connectionError) {
      return connectionError;
    }
    return fieldTypeName;
  }, [connectionError, fieldTypeName, isConnectionInProgress]);

  return (
    <Tooltip
      label={tooltip}
      placement={handleType === 'target' ? 'start' : 'end'}
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

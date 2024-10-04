import { Tooltip } from '@invoke-ai/ui-library';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { getFieldColor } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import { useFieldTypeName } from 'features/nodes/hooks/usePrettyFieldType';
import type { ValidationResult } from 'features/nodes/store/util/validateConnection';
import { HANDLE_TOOLTIP_OPEN_DELAY, MODEL_TYPES } from 'features/nodes/types/constants';
import { type FieldInputTemplate, type FieldOutputTemplate, isSingle } from 'features/nodes/types/field';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { HandleType } from 'reactflow';
import { Handle, Position } from 'reactflow';

type FieldHandleProps = {
  fieldTemplate: FieldInputTemplate | FieldOutputTemplate;
  handleType: HandleType;
  isConnectionInProgress: boolean;
  isConnectionStartField: boolean;
  validationResult: ValidationResult;
};

const FieldHandle = (props: FieldHandleProps) => {
  const { fieldTemplate, handleType, isConnectionInProgress, isConnectionStartField, validationResult } = props;
  const { t } = useTranslation();
  const { name } = fieldTemplate;
  const type = fieldTemplate.type;
  const fieldTypeName = useFieldTypeName(type);
  const styles: CSSProperties = useMemo(() => {
    const isModelType = MODEL_TYPES.some((t) => t === type.name);
    const color = getFieldColor(type);
    const s: CSSProperties = {
      backgroundColor: !isSingle(type) ? colorTokenToCssVar('base.900') : color,
      position: 'absolute',
      width: '1rem',
      height: '1rem',
      borderWidth: !isSingle(type) ? 4 : 0,
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

    if (isConnectionInProgress && !isConnectionStartField && !validationResult.isValid) {
      s.filter = 'opacity(0.4) grayscale(0.7)';
    }

    if (isConnectionInProgress && !validationResult.isValid) {
      if (isConnectionStartField) {
        s.cursor = 'grab';
      } else {
        s.cursor = 'not-allowed';
      }
    } else {
      s.cursor = 'crosshair';
    }

    return s;
  }, [handleType, isConnectionInProgress, isConnectionStartField, type, validationResult.isValid]);

  const tooltip = useMemo(() => {
    if (isConnectionInProgress && validationResult.messageTKey) {
      return t(validationResult.messageTKey);
    }
    return fieldTypeName;
  }, [fieldTypeName, isConnectionInProgress, t, validationResult.messageTKey]);

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

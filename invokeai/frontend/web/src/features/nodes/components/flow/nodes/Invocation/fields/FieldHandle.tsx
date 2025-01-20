import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Tooltip } from '@invoke-ai/ui-library';
import { getFieldColor } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import { useFieldTypeName } from 'features/nodes/hooks/usePrettyFieldType';
import type { ValidationResult } from 'features/nodes/store/util/validateConnection';
import { HANDLE_TOOLTIP_OPEN_DELAY, MODEL_TYPES } from 'features/nodes/types/constants';
import type { FieldInputTemplate, FieldOutputTemplate } from 'features/nodes/types/field';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { HandleType } from 'reactflow';
import { Handle, Position } from 'reactflow';

type Props = {
  handleType: HandleType;
  fieldTemplate: FieldInputTemplate | FieldOutputTemplate;
  isConnectionInProgress: boolean;
  isConnectionStartField: boolean;
  validationResult: ValidationResult;
};

const sx = {
  position: 'relative',
  width: 'full',
  height: 'full',
  borderStyle: 'solid',
  borderWidth: 4,
  pointerEvents: 'none',
  '&[data-cardinality="SINGLE"]': {
    borderWidth: 0,
  },
  borderRadius: '100%',
  '&[data-is-model-field="true"], &[data-is-batch-field="true"]': {
    borderRadius: 4,
  },
  '&[data-is-batch-field="true"]': {
    transform: 'rotate(45deg)',
  },
  '&[data-is-connection-in-progress="true"][data-is-connection-start-field="false"][data-is-connection-valid="false"]':
    {
      filter: 'opacity(0.4) grayscale(0.7)',
      cursor: 'not-allowed',
    },
  '&[data-is-connection-in-progress="true"][data-is-connection-start-field="true"][data-is-connection-valid="false"]': {
    cursor: 'grab',
  },
  '&[data-is-connection-in-progress="false"] &[data-is-connection-valid="true"]': {
    cursor: 'crosshair',
  },
} satisfies SystemStyleObject;

const handleStyleBase = {
  position: 'absolute',
  width: '1rem',
  height: '1rem',
  zIndex: 1,
  background: 'none',
  border: 'none',
} satisfies CSSProperties;

const targetHandleStyle = {
  ...handleStyleBase,
  insetInlineStart: '-1rem',
} satisfies CSSProperties;

const sourceHandleStyle = {
  ...handleStyleBase,
  insetInlineEnd: '-1rem',
} satisfies CSSProperties;

export const FieldHandle = memo((props: Props) => {
  const { fieldTemplate, isConnectionInProgress, isConnectionStartField, validationResult, handleType } = props;
  const { t } = useTranslation();
  const fieldTypeName = useFieldTypeName(fieldTemplate.type);
  const fieldColor = useMemo(() => getFieldColor(fieldTemplate.type), [fieldTemplate.type]);
  const isModelField = useMemo(() => MODEL_TYPES.some((t) => t === fieldTemplate.type.name), [fieldTemplate.type]);

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
        id={fieldTemplate.name}
        position={handleType === 'target' ? Position.Left : Position.Right}
        style={handleType === 'target' ? targetHandleStyle : sourceHandleStyle}
      >
        <Box
          sx={sx}
          data-cardinality={fieldTemplate.type.cardinality}
          data-is-batch-field={fieldTemplate.type.batch}
          data-is-model-field={isModelField}
          data-is-connection-in-progress={isConnectionInProgress}
          data-is-connection-start-field={isConnectionStartField}
          data-is-connection-valid={validationResult.isValid}
          backgroundColor={fieldTemplate.type.cardinality === 'SINGLE' ? fieldColor : 'base.900'}
          borderColor={fieldColor}
        />
      </Handle>
    </Tooltip>
  );
});

FieldHandle.displayName = 'FieldHandle';

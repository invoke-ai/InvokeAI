import { Box, Tooltip } from '@invoke-ai/ui-library';
import { Handle, Position } from '@xyflow/react';
import { getFieldColor } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import {
  NODE_IO_HANDLE_HITBOX_INPUT,
  NODE_IO_HANDLE_INNER_SX,
} from 'features/nodes/components/flow/nodes/common/nodeIOHandle';
import {
  useConnectionErrorTKey,
  useIsConnectionInProgress,
  useIsConnectionStartField,
} from 'features/nodes/hooks/useFieldConnectionState';
import { useInputFieldTemplateOrThrow } from 'features/nodes/hooks/useInputFieldTemplateOrThrow';
import { useFieldTypeName } from 'features/nodes/hooks/usePrettyFieldType';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { isModelFieldType } from 'features/nodes/types/field';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldHandle = memo(({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useInputFieldTemplateOrThrow(fieldName);
  const fieldTypeName = useFieldTypeName(fieldTemplate.type);
  const fieldColor = useMemo(() => getFieldColor(fieldTemplate.type), [fieldTemplate.type]);
  const isModelField = useMemo(() => isModelFieldType(fieldTemplate.type), [fieldTemplate.type]);
  const isConnectionInProgress = useIsConnectionInProgress();

  if (isConnectionInProgress) {
    return (
      <ConnectionInProgressHandle
        nodeId={nodeId}
        fieldName={fieldName}
        fieldTemplate={fieldTemplate}
        fieldTypeName={fieldTypeName}
        fieldColor={fieldColor}
        isModelField={isModelField}
      />
    );
  }

  return (
    <IdleHandle
      nodeId={nodeId}
      fieldName={fieldName}
      fieldTemplate={fieldTemplate}
      fieldTypeName={fieldTypeName}
      fieldColor={fieldColor}
      isModelField={isModelField}
    />
  );
});

InputFieldHandle.displayName = 'InputFieldHandle';

type HandleCommonProps = {
  nodeId: string;
  fieldName: string;
  fieldTemplate: FieldInputTemplate;
  fieldTypeName: string;
  fieldColor: string;
  isModelField: boolean;
};

const IdleHandle = memo(({ fieldTemplate, fieldTypeName, fieldColor, isModelField }: HandleCommonProps) => {
  return (
    <Tooltip label={fieldTypeName} placement="start" openDelay={HANDLE_TOOLTIP_OPEN_DELAY}>
      <Handle type="target" id={fieldTemplate.name} position={Position.Left} style={NODE_IO_HANDLE_HITBOX_INPUT}>
        <Box
          sx={NODE_IO_HANDLE_INNER_SX}
          data-cardinality={fieldTemplate.type.cardinality}
          data-is-batch-field={fieldTemplate.type.batch}
          data-is-model-field={isModelField}
          data-is-connection-in-progress={false}
          data-is-connection-start-field={false}
          data-is-connection-valid={false}
          backgroundColor={fieldTemplate.type.cardinality === 'SINGLE' ? fieldColor : 'base.900'}
          borderColor={fieldColor}
        />
      </Handle>
    </Tooltip>
  );
});
IdleHandle.displayName = 'IdleHandle';

const ConnectionInProgressHandle = memo(
  ({ nodeId, fieldName, fieldTemplate, fieldTypeName, fieldColor, isModelField }: HandleCommonProps) => {
    const { t } = useTranslation();
    const isConnectionStartField = useIsConnectionStartField(nodeId, fieldName, 'target');
    const connectionError = useConnectionErrorTKey(nodeId, fieldName, 'target');

    const tooltip = useMemo(() => {
      if (connectionError !== null) {
        return t(connectionError);
      }
      return fieldTypeName;
    }, [fieldTypeName, t, connectionError]);

    return (
      <Tooltip label={tooltip} placement="start" openDelay={HANDLE_TOOLTIP_OPEN_DELAY}>
        <Handle type="target" id={fieldTemplate.name} position={Position.Left} style={NODE_IO_HANDLE_HITBOX_INPUT}>
          <Box
            sx={NODE_IO_HANDLE_INNER_SX}
            data-cardinality={fieldTemplate.type.cardinality}
            data-is-batch-field={fieldTemplate.type.batch}
            data-is-model-field={isModelField}
            data-is-connection-in-progress={true}
            data-is-connection-start-field={isConnectionStartField}
            data-is-connection-valid={connectionError === null}
            backgroundColor={fieldTemplate.type.cardinality === 'SINGLE' ? fieldColor : 'base.900'}
            borderColor={fieldColor}
          />
        </Handle>
      </Tooltip>
    );
  }
);
ConnectionInProgressHandle.displayName = 'ConnectionInProgressHandle';

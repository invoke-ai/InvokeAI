import { Box, Icon, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { Handle, type HandleType, type Node, type NodeProps, Position } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import { getFieldColor } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import {
  NODE_IO_HANDLE_HITBOX_INPUT,
  NODE_IO_HANDLE_HITBOX_OUTPUT,
  NODE_IO_HANDLE_INNER_SX,
} from 'features/nodes/components/flow/nodes/common/nodeIOHandle';
import NonInvocationNodeWrapper from 'features/nodes/components/flow/nodes/common/NonInvocationNodeWrapper';
import {
  useConnectionErrorTKey,
  useIsConnectionInProgress,
  useIsConnectionStartField,
} from 'features/nodes/hooks/useFieldConnectionState';
import { useFieldTypeName } from 'features/nodes/hooks/usePrettyFieldType';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectEdges, selectNodes } from 'features/nodes/store/selectors';
import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  resolveConnectorDisplayFieldType,
} from 'features/nodes/store/util/connectorTopology';
import { HANDLE_TOOLTIP_OPEN_DELAY, NO_DRAG_CLASS } from 'features/nodes/types/constants';
import type { FieldType } from 'features/nodes/types/field';
import { isModelFieldType } from 'features/nodes/types/field';
import type { ConnectorNodeData } from 'features/nodes/types/invocation';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotOutlineFill } from 'react-icons/pi';

const CONNECTOR_NODE_SIZE = 35;

/** AnyField-shaped fallback for tooltips when display type is unknown; same shape as connector stubs in `useConnection`. */
const CONNECTOR_FALLBACK_FIELD_TYPE = {
  name: 'AnyField',
  cardinality: 'SINGLE',
  batch: false,
} as const satisfies FieldType;

const CONNECTOR_HANDLE_VERTICAL_ALIGN: CSSProperties = {
  top: '50%',
  transform: 'translateY(-50%)',
};

const CONNECTOR_HANDLE_INPUT_STYLE = {
  ...NODE_IO_HANDLE_HITBOX_INPUT,
  ...CONNECTOR_HANDLE_VERTICAL_ALIGN,
} satisfies CSSProperties;

const CONNECTOR_HANDLE_OUTPUT_STYLE = {
  ...NODE_IO_HANDLE_HITBOX_OUTPUT,
  ...CONNECTOR_HANDLE_VERTICAL_ALIGN,
} satisfies CSSProperties;

type PassthroughHandleProps = {
  nodeId: string;
  rfHandleType: HandleType;
  handleId: string;
  position: Position;
  hitboxStyle: CSSProperties;
  displayFieldType: FieldType | null;
  fieldColor: string;
  fieldTypeName: string;
};

const ConnectorPassthroughHandle = memo(
  ({
    nodeId,
    rfHandleType,
    handleId,
    position,
    hitboxStyle,
    displayFieldType,
    fieldColor,
    fieldTypeName,
  }: PassthroughHandleProps) => {
    const { t } = useTranslation();
    const isConnectionInProgress = useIsConnectionInProgress();
    const isConnectionStartField = useIsConnectionStartField(nodeId, handleId, rfHandleType);
    const connectionError = useConnectionErrorTKey(nodeId, handleId, rfHandleType);

    const tooltipLabel = useMemo(() => {
      if (isConnectionInProgress && connectionError !== null) {
        return t(connectionError);
      }
      return fieldTypeName;
    }, [connectionError, fieldTypeName, isConnectionInProgress, t]);

    const innerProps = useMemo(() => {
      const shape =
        displayFieldType !== null
          ? {
              'data-cardinality': displayFieldType.cardinality,
              'data-is-batch-field': displayFieldType.batch,
              'data-is-model-field': isModelFieldType(displayFieldType),
            }
          : {
              'data-cardinality': 'SINGLE' as const,
              'data-is-batch-field': false,
              'data-is-model-field': false,
            };

      return {
        sx: NODE_IO_HANDLE_INNER_SX,
        ...shape,
        'data-is-connection-in-progress': isConnectionInProgress,
        'data-is-connection-start-field': isConnectionInProgress ? isConnectionStartField : false,
        'data-is-connection-valid': isConnectionInProgress ? connectionError === null : false,
      };
    }, [connectionError, displayFieldType, isConnectionInProgress, isConnectionStartField]);

    const innerBackgroundColor =
      displayFieldType !== null && displayFieldType.cardinality !== 'SINGLE' ? 'base.900' : fieldColor;

    return (
      <Tooltip label={tooltipLabel} placement="start" openDelay={HANDLE_TOOLTIP_OPEN_DELAY}>
        <Handle className={NO_DRAG_CLASS} type={rfHandleType} id={handleId} position={position} style={hitboxStyle}>
          <Box {...innerProps} backgroundColor={innerBackgroundColor} borderColor={fieldColor} />
        </Handle>
      </Tooltip>
    );
  }
);

ConnectorPassthroughHandle.displayName = 'ConnectorPassthroughHandle';

const ConnectorNode = ({ id, selected }: NodeProps<Node<ConnectorNodeData>>) => {
  const templates = useStore($templates);
  const nodes = useAppSelector(selectNodes);
  const edges = useAppSelector(selectEdges);

  const displayFieldType = useMemo(
    () => resolveConnectorDisplayFieldType(id, nodes, edges, templates),
    [id, nodes, edges, templates]
  );

  const fieldColor = useMemo(() => getFieldColor(displayFieldType), [displayFieldType]);

  const fieldTypeLabel = useFieldTypeName(displayFieldType ?? CONNECTOR_FALLBACK_FIELD_TYPE);

  return (
    <NonInvocationNodeWrapper nodeId={id} selected={selected} width={CONNECTOR_NODE_SIZE} borderRadius="full">
      <Box
        data-connector-node-context-menu="true"
        data-connector-node-id={id}
        position="relative"
        w={CONNECTOR_NODE_SIZE}
        h={CONNECTOR_NODE_SIZE}
        display="flex"
        alignItems="center"
        justifyContent="center"
      >
        <ConnectorPassthroughHandle
          nodeId={id}
          rfHandleType="target"
          handleId={CONNECTOR_INPUT_HANDLE}
          position={Position.Left}
          hitboxStyle={CONNECTOR_HANDLE_INPUT_STYLE}
          displayFieldType={displayFieldType}
          fieldColor={fieldColor}
          fieldTypeName={fieldTypeLabel}
        />
        <Box
          layerStyle="nodeBody"
          position="relative"
          w={CONNECTOR_NODE_SIZE}
          h={CONNECTOR_NODE_SIZE}
          display="flex"
          alignItems="center"
          justifyContent="center"
          borderRadius="full"
        >
          <Icon as={PiDotOutlineFill} boxSize={5} color="base.300" />
        </Box>
        <ConnectorPassthroughHandle
          nodeId={id}
          rfHandleType="source"
          handleId={CONNECTOR_OUTPUT_HANDLE}
          position={Position.Right}
          hitboxStyle={CONNECTOR_HANDLE_OUTPUT_STYLE}
          displayFieldType={displayFieldType}
          fieldColor={fieldColor}
          fieldTypeName={fieldTypeLabel}
        />
      </Box>
    </NonInvocationNodeWrapper>
  );
};

export default memo(ConnectorNode);

import type { FieldType } from '@features/workflow/contracts';

import { Box, Flex } from '@chakra-ui/react';
import {
  getWorkflowNodeChromeProps,
  getWorkflowNodeHandleStyle,
  WORKFLOW_NODE_HANDLE_SIZE,
} from '@features/workflow/ui/nodeChrome';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE, getFieldTypeLabel } from '@features/workflow/utility';
import { Tooltip } from '@platform/ui';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { memo, useMemo } from 'react';

import type { ConnectorFlowNode as ConnectorFlowNodeType } from './flowAdapters';

import { getHandleTypeTooltip } from './handleTooltip';

/** An untyped ("any") connector end: a plain grid-tinted dot rather than a field-typed handle. */
const genericHandleStyle: React.CSSProperties = {
  background: 'var(--wb-flow-grid)',
  border: '1px solid var(--xy-background-color)',
  height: WORKFLOW_NODE_HANDLE_SIZE,
  width: WORKFLOW_NODE_HANDLE_SIZE,
};

const getConnectorHandleStyle = (type: FieldType | null, side: 'left' | 'right'): React.CSSProperties =>
  type ? getWorkflowNodeHandleStyle(type, side) : genericHandleStyle;

const getConnectorTitle = (inputType: FieldType | null, outputType: FieldType | null): string => {
  const inputLabel = inputType ? getFieldTypeLabel(inputType) : 'Any input';
  const outputLabel = outputType ? getFieldTypeLabel(outputType) : 'Any output';

  return `Connector: ${inputLabel} -> ${outputLabel}`;
};

const ConnectorFlowNodeComponent = ({ data, selected }: NodeProps<ConnectorFlowNodeType>) => {
  const node = data.documentNode;
  const inputHandleStyle = useMemo(
    () => ({ ...getConnectorHandleStyle(data.inputFieldType, 'left'), left: -WORKFLOW_NODE_HANDLE_SIZE / 2 }),
    [data.inputFieldType]
  );
  const outputHandleStyle = useMemo(
    () => ({ ...getConnectorHandleStyle(data.outputFieldType, 'right'), right: -WORKFLOW_NODE_HANDLE_SIZE / 2 }),
    [data.outputFieldType]
  );

  return (
    <Flex align="center" data-connector-node-id={node.id} justify="center" position="relative">
      <Tooltip content={getHandleTypeTooltip(data.inputFieldType, 'Any input')} showArrow>
        <Handle id={CONNECTOR_INPUT_HANDLE} position={Position.Left} style={inputHandleStyle} type="target" />
      </Tooltip>
      <Tooltip content={getConnectorTitle(data.inputFieldType, data.outputFieldType)} showArrow>
        <Box bg="bg" h="1rem" rounded="full" w="2.5rem" {...getWorkflowNodeChromeProps({ selected })} />
      </Tooltip>
      <Tooltip content={getHandleTypeTooltip(data.outputFieldType, 'Any output')} showArrow>
        <Handle id={CONNECTOR_OUTPUT_HANDLE} position={Position.Right} style={outputHandleStyle} type="source" />
      </Tooltip>
    </Flex>
  );
};

export const ConnectorFlowNode = memo(ConnectorFlowNodeComponent);

import type { FieldType } from '@workbench/workflows/types';

import { Box, Flex } from '@chakra-ui/react';
import { Tooltip } from '@workbench/components/ui/Tooltip';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from '@workbench/workflows/connectors';
import { getFieldTypeColor, getFieldTypeLabel, isModelFieldType } from '@workbench/workflows/fields';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { memo } from 'react';

import type { ConnectorFlowNode as ConnectorFlowNodeType } from './flowAdapters';

import { getHandleTypeTooltip } from './handleTooltip';

const HANDLE_SIZE = 12;
type HandleSide = 'left' | 'right';

const genericHandleStyle: React.CSSProperties = {
  background: 'var(--wb-flow-grid)',
  border: '1px solid var(--chakra-colors-bg)',
  height: HANDLE_SIZE,
  width: HANDLE_SIZE,
};

const typedHandleStyle = (type: FieldType, side: HandleSide): React.CSSProperties => {
  const color = getFieldTypeColor(type);
  const isFilled = type.cardinality === 'SINGLE';
  const isAngular = isModelFieldType(type) || type.batch;
  const centeredDiamondTransform = `translate(${side === 'left' ? '-' : ''}50%, -50%) rotate(45deg)`;

  return {
    background: isFilled ? color : 'var(--xy-background-color)',
    border: isFilled ? 'none' : `3px solid ${color}`,
    borderRadius: isAngular ? 3 : '50%',
    boxShadow: '0 0 0 1px var(--xy-background-color)',
    height: HANDLE_SIZE,
    transform: type.batch ? centeredDiamondTransform : undefined,
    width: HANDLE_SIZE,
  };
};

const getConnectorHandleStyle = (type: FieldType | null, side: HandleSide): React.CSSProperties =>
  type ? typedHandleStyle(type, side) : genericHandleStyle;

const getConnectorTitle = (inputType: FieldType | null, outputType: FieldType | null): string => {
  const inputLabel = inputType ? getFieldTypeLabel(inputType) : 'Any input';
  const outputLabel = outputType ? getFieldTypeLabel(outputType) : 'Any output';

  return `Connector: ${inputLabel} -> ${outputLabel}`;
};

const ConnectorFlowNodeComponent = ({ data, selected }: NodeProps<ConnectorFlowNodeType>) => {
  const node = data.documentNode;

  return (
    <Flex align="center" data-connector-node-id={node.id} justify="center" position="relative">
      <Tooltip content={getHandleTypeTooltip(data.inputFieldType, 'Any input')} showArrow>
        <Handle
          id={CONNECTOR_INPUT_HANDLE}
          position={Position.Left}
          style={{ ...getConnectorHandleStyle(data.inputFieldType, 'left'), left: -HANDLE_SIZE / 2 }}
          type="target"
        />
      </Tooltip>
      <Tooltip content={getConnectorTitle(data.inputFieldType, data.outputFieldType)} showArrow>
        <Box
          bg="bg"
          borderColor={selected ? 'accent.solid' : 'border.emphasized'}
          borderWidth="1px"
          h="1rem"
          rounded="full"
          shadow={selected ? 'md' : 'sm'}
          transition="border-color 0.12s ease, box-shadow 0.12s ease"
          w="2.5rem"
          _hover={selected ? undefined : { borderColor: 'brand.solid', shadow: 'md' }}
        />
      </Tooltip>
      <Tooltip content={getHandleTypeTooltip(data.outputFieldType, 'Any output')} showArrow>
        <Handle
          id={CONNECTOR_OUTPUT_HANDLE}
          position={Position.Right}
          style={{ ...getConnectorHandleStyle(data.outputFieldType, 'right'), right: -HANDLE_SIZE / 2 }}
          type="source"
        />
      </Tooltip>
    </Flex>
  );
};

export const ConnectorFlowNode = memo(ConnectorFlowNodeComponent);

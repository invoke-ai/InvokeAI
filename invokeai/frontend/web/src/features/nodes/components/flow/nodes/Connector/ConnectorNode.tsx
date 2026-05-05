import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Icon } from '@invoke-ai/ui-library';
import type { Node, NodeProps } from '@xyflow/react';
import { Handle, Position } from '@xyflow/react';
import NonInvocationNodeWrapper from 'features/nodes/components/flow/nodes/common/NonInvocationNodeWrapper';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from 'features/nodes/store/util/connectorTopology';
import { NO_DRAG_CLASS } from 'features/nodes/types/constants';
import type { ConnectorNodeData } from 'features/nodes/types/invocation';
import type { CSSProperties } from 'react';
import { memo } from 'react';
import { PiDotOutlineFill } from 'react-icons/pi';

const CONNECTOR_NODE_SIZE = 35;
const CONNECTOR_HANDLE_SIZE = 24;
const CONNECTOR_HANDLE_OFFSET = -CONNECTOR_HANDLE_SIZE / 2;

const handleVisualSx = {
  position: 'absolute',
  top: '50%',
  w: 4,
  h: 4,
  borderRadius: 'full',
  borderWidth: 2,
  borderColor: 'base.900',
  bg: 'base.100',
  pointerEvents: 'none',
} satisfies SystemStyleObject;

const inputHandleVisualSx = {
  ...handleVisualSx,
  left: 0,
  transform: 'translate(-50%, -50%)',
} satisfies SystemStyleObject;

const outputHandleVisualSx = {
  ...handleVisualSx,
  right: 0,
  transform: 'translate(50%, -50%)',
} satisfies SystemStyleObject;

const connectorSx = {
  '& .connector-border': {
    pointerEvents: 'none',
    position: 'absolute',
    inset: 0,
    borderRadius: 'inherit',
    shadow: '0 0 0 1px var(--invoke-colors-base-500)',
  },
  _hover: {
    '& .connector-border': {
      shadow: '0 0 0 1px var(--invoke-colors-blue-300)',
    },
    '&[data-is-selected="true"] .connector-border': {
      shadow: '0 0 0 2px var(--invoke-colors-blue-300)',
    },
  },
  '&[data-is-selected="true"] .connector-border': {
    shadow: '0 0 0 2px var(--invoke-colors-blue-300)',
  },
} satisfies SystemStyleObject;

const handleStyles = {
  position: 'absolute',
  width: `${CONNECTOR_HANDLE_SIZE}px`,
  height: `${CONNECTOR_HANDLE_SIZE}px`,
  top: `calc(50% + ${CONNECTOR_HANDLE_OFFSET}px)`,
  zIndex: 1,
  background: 'none',
  border: 'none',
} satisfies CSSProperties;

const inputHandleStyles = {
  ...handleStyles,
  left: 0,
  transform: 'none',
} satisfies CSSProperties;

const outputHandleStyles = {
  ...handleStyles,
  right: 0,
  transform: 'none',
} satisfies CSSProperties;

const ConnectorNode = ({ id, selected }: NodeProps<Node<ConnectorNodeData>>) => {
  return (
    <NonInvocationNodeWrapper
      nodeId={id}
      selected={selected}
      width={CONNECTOR_NODE_SIZE}
      borderRadius="full"
      withChrome={false}
    >
      <Box
        data-connector-node-context-menu="true"
        data-connector-node-id={id}
        data-is-selected={selected}
        position="relative"
        w={CONNECTOR_NODE_SIZE}
        h={CONNECTOR_NODE_SIZE}
        display="flex"
        alignItems="center"
        justifyContent="center"
        sx={connectorSx}
      >
        <Handle
          className={NO_DRAG_CLASS}
          type="target"
          id={CONNECTOR_INPUT_HANDLE}
          position={Position.Left}
          style={inputHandleStyles}
        >
          <Box sx={inputHandleVisualSx} />
        </Handle>
        <Box
          position="relative"
          w={CONNECTOR_NODE_SIZE}
          h={CONNECTOR_NODE_SIZE}
          display="flex"
          alignItems="center"
          justifyContent="center"
          borderRadius="full"
          bg={selected ? 'base.650' : 'base.700'}
        >
          <Box className="connector-border" />
          <Icon as={PiDotOutlineFill} boxSize={5} color={selected ? 'base.50' : 'base.100'} />
        </Box>
        <Handle
          className={NO_DRAG_CLASS}
          type="source"
          id={CONNECTOR_OUTPUT_HANDLE}
          position={Position.Right}
          style={outputHandleStyles}
        >
          <Box sx={outputHandleVisualSx} />
        </Handle>
      </Box>
    </NonInvocationNodeWrapper>
  );
};

export default memo(ConnectorNode);

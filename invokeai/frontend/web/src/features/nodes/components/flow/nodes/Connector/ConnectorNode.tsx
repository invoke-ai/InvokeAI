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

const handleVisualSx = {
  w: 3,
  h: 3,
  borderRadius: 'full',
  borderWidth: 2,
  borderColor: 'base.900',
  bg: 'base.100',
  pointerEvents: 'none',
} satisfies SystemStyleObject;

const handleStyles = {
  position: 'absolute',
  width: '1rem',
  height: '1rem',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  top: '50%',
  transform: 'translateY(-50%)',
  zIndex: 1,
  background: 'none',
  border: 'none',
} satisfies CSSProperties;

const inputHandleStyles = {
  ...handleStyles,
  insetInlineStart: 0,
  justifyContent: 'flex-start',
} satisfies CSSProperties;

const outputHandleStyles = {
  ...handleStyles,
  insetInlineEnd: 0,
  justifyContent: 'flex-end',
} satisfies CSSProperties;

const ConnectorNode = ({ id, selected }: NodeProps<Node<ConnectorNodeData>>) => {
  return (
    <NonInvocationNodeWrapper nodeId={id} selected={selected} width={25} borderRadius="full" withChrome={false}>
      <Box
        data-connector-node-context-menu="true"
        data-connector-node-id={id}
        position="relative"
        w={25}
        h={25}
        display="flex"
        alignItems="center"
        justifyContent="center"
      >
        <Handle
          className={NO_DRAG_CLASS}
          type="target"
          id={CONNECTOR_INPUT_HANDLE}
          position={Position.Left}
          style={inputHandleStyles}
        >
          <Box sx={handleVisualSx} />
        </Handle>
        <Box
          w={13}
          h={13}
          display="flex"
          alignItems="center"
          justifyContent="center"
          borderRadius="full"
          bg={selected ? 'base.650' : 'base.700'}
          boxShadow={selected ? '0 0 0 2px var(--invoke-colors-blue-300)' : '0 0 0 1px var(--invoke-colors-base-500)'}
        >
          <Icon as={PiDotOutlineFill} boxSize={5} color={selected ? 'base.50' : 'base.100'} />
        </Box>
        <Handle
          className={NO_DRAG_CLASS}
          type="source"
          id={CONNECTOR_OUTPUT_HANDLE}
          position={Position.Right}
          style={outputHandleStyles}
        >
          <Box sx={handleVisualSx} />
        </Handle>
      </Box>
    </NonInvocationNodeWrapper>
  );
};

export default memo(ConnectorNode);

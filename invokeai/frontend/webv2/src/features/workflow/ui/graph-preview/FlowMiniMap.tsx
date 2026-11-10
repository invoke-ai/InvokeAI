import type { ComponentProps } from 'react';

import { chakra } from '@chakra-ui/react';
import { MiniMap } from '@xyflow/react';

/** Style the MiniMap frame with Chakra tokens and its SVG fills with flowTheme variables. */
const StyledMiniMap = chakra(MiniMap);

/** Minimap node classes a caller may set; each tints the node in a state colour. */
export const FLOW_MINIMAP_ERROR_CLASS = 'workflow-minimap-error';
export const FLOW_MINIMAP_RUNNING_CLASS = 'workflow-minimap-running';

const MINIMAP_STATE_CSS = {
  [`& .react-flow__minimap-node.${FLOW_MINIMAP_ERROR_CLASS}`]: { fill: '{colors.border.error}' },
  [`& .react-flow__minimap-node.${FLOW_MINIMAP_RUNNING_CLASS}`]: { fill: '{colors.accent.solid}' },
};

export const FlowMiniMap = ({ nodeClassName }: Pick<ComponentProps<typeof MiniMap>, 'nodeClassName'>) => (
  <StyledMiniMap
    borderColor="border.subtle"
    borderWidth="1px"
    css={MINIMAP_STATE_CSS}
    m="3"
    nodeClassName={nodeClassName}
    overflow="hidden"
    pannable
    position="bottom-right"
    rounded="md"
    shadow="sm"
    zoomable
  />
);

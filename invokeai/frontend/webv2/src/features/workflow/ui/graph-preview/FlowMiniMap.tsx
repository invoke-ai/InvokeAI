import { chakra } from '@chakra-ui/react';
import { MiniMap } from '@xyflow/react';

/**
 * xyflow's MiniMap wrapped in the chakra factory so its frame is styled with
 * workbench tokens (the fills inside the SVG come from the `--xy-minimap-*`
 * vars in `flowTheme.ts`).
 */
const StyledMiniMap = chakra(MiniMap);

export const FlowMiniMap = () => (
  <StyledMiniMap
    borderColor="border.subtle"
    borderWidth="1px"
    m="3"
    overflow="hidden"
    pannable
    position="bottom-right"
    rounded="md"
    shadow="sm"
    zoomable
  />
);

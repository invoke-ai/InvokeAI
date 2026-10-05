import { chakra } from '@chakra-ui/react';
import { MiniMap } from '@xyflow/react';

/** Style the MiniMap frame with Chakra tokens and its SVG fills with flowTheme variables. */
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

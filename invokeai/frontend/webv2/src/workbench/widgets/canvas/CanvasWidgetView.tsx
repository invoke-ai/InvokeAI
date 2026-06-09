import { Box } from '@chakra-ui/react';

import { GraphBearingWidgetHeader } from '../../components/WidgetFrames';
import { WidgetFailureBoundary } from '../../components/WidgetFailureBoundary';
import type { WidgetViewProps } from '../../types';

export const CanvasWidgetView = ({ manifest }: WidgetViewProps) => (
  <WidgetFailureBoundary widgetId="canvas">
    <Box
      bg="bg.canvas"
      h="full"
      overflow="hidden"
      position="relative"
      w="full"
      bgImage="radial-gradient(var(--chakra-colors-canvas-dot) 1.5px, transparent 1.5px)"
      bgSize="28px 28px"
    >
      <Box position="absolute" right="2" top="2" zIndex="2">
        <GraphBearingWidgetHeader manifest={manifest} region="center" />
      </Box>
      <ToolScrubber />
      <StagingFrame top="120px" left="56px" w="62%" h="56%" isPrimary />
      <StagingFrame top="8px" right="0" w="52%" h="104px" />
      <StagingFrame bottom="-12px" right="12%" w="52%" h="116px" />
    </Box>
  </WidgetFailureBoundary>
);

interface StagingFrameProps {
  isPrimary?: boolean;
  top?: string;
  left?: string;
  right?: string;
  bottom?: string;
  w: string;
  h: string;
}

const StagingFrame = ({ isPrimary, ...position }: StagingFrameProps) => (
  <Box
    bg="bg.panel"
    boxShadow="inset 0 0 0 1px var(--chakra-colors-border-panel)"
    p={isPrimary ? '3' : undefined}
    position="absolute"
    rounded="sm"
    {...position}
  />
);

const ToolScrubber = () => (
  <Box
    bg="bg.shell"
    borderWidth="1px"
    borderColor="border.emphasis"
    h="22rem"
    left="2.5"
    position="absolute"
    rounded="md"
    top="2.5"
    w="8"
    zIndex="1"
  />
);

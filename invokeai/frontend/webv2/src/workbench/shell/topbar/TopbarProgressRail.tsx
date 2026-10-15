import type { SystemStyleObject } from '@chakra-ui/react';

import { QueueProgressRail } from '@workbench/queue-integration/QueueProgressRail';

/** Overlay the shared progress rail on the divider for visibility in every layout without reflow. */

const RAIL_SX: SystemStyleObject = {
  bottom: '-1px',
  display: 'flex',
  gap: '1px',
  height: '2px',
  insetInline: 0,
  pointerEvents: 'none',
  position: 'absolute',
  zIndex: 3,
};

export const TopbarProgressRail = () => <QueueProgressRail css={RAIL_SX} />;

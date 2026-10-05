import type { WidgetRegionDropState } from '@workbench/widgetDnd';

import { DropZone, type DropZoneProps } from '@platform/ui/DropZone';

const DISALLOWED_STYLES: DropZoneProps = { bg: 'bg.muted', borderColor: 'border.subtle' };

/** Share DropZone styling across full rails and inset status clusters; consumers supply placement only. */
export const WidgetRegionDropOverlay = ({
  dropState,
  isOver,
  ...boxProps
}: {
  dropState: WidgetRegionDropState;
  isOver: boolean;
} & DropZoneProps) => (
  <DropZone
    bottom="0"
    isOver={dropState.isAllowed && isOver}
    left="0"
    opacity={dropState.isAllowed ? 0.96 : 0.5}
    pointerEvents="none"
    position="absolute"
    right="0"
    top="0"
    variant="overlay"
    zIndex="2"
    {...(dropState.isAllowed ? undefined : DISALLOWED_STYLES)}
    {...boxProps}
  />
);

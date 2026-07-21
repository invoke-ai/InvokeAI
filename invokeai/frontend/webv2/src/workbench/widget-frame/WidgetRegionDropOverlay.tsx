import type { WidgetRegionDropState } from '@workbench/widgetDnd';

import { DropZone, type DropZoneProps } from '@platform/ui';

const DISALLOWED_STYLES: DropZoneProps = { bg: 'bg.muted', borderColor: 'border.subtle' };

export const WidgetRegionDropOverlay = ({
  dropState,
  isOver,
}: {
  dropState: WidgetRegionDropState;
  isOver: boolean;
}) => (
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
  />
);

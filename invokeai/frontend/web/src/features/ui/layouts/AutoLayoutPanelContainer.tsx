import { Box, type SystemStyleObject } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import type { IDockviewPanelProps, IGridviewPanelProps } from 'dockview';
import { selectSystemShouldEnableHighlightFocusedRegions } from 'features/system/store/systemSlice';
import type { PanelParameters } from 'features/ui/layouts/auto-layout-context';
import type { PropsWithChildren } from 'react';
import { memo, useRef } from 'react';

const sx: SystemStyleObject = {
  position: 'relative',
  w: 'full',
  h: 'full',
  p: 2,
  '&[data-highlighted="true"]::after': {
    borderColor: 'invokeBlue.300',
  },
  '&::after': {
    content: '""',
    position: 'absolute',
    inset: '2px',
    zIndex: 1,
    borderRadius: 'base',
    border: '1px solid',
    borderColor: 'transparent',
    pointerEvents: 'none',
  },
};

export const AutoLayoutPanelContainer = memo(
  (
    props:
      | PropsWithChildren<IDockviewPanelProps<PanelParameters>>
      | PropsWithChildren<IGridviewPanelProps<PanelParameters>>
  ) => {
    const ref = useRef<HTMLDivElement>(null);
    const shouldHighlightFocusedRegions = useAppSelector(selectSystemShouldEnableHighlightFocusedRegions);

    useFocusRegion(props.params.focusRegion, ref);

    const isFocused = useIsRegionFocused(props.params.focusRegion);
    const isHighlighted = isFocused && shouldHighlightFocusedRegions;

    return (
      <Box ref={ref} tabIndex={-1} sx={sx} data-highlighted={isHighlighted}>
        {props.children}
      </Box>
    );
  }
);
AutoLayoutPanelContainer.displayName = 'AutoLayoutPanelContainer';

import { Box, type BoxProps, type SystemStyleObject } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { type FocusRegionName, useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import type { IDockviewPanelProps, IGridviewPanelProps } from 'dockview';
import { selectSystemShouldEnableHighlightFocusedRegions } from 'features/system/store/systemSlice';
import type { PanelParameters } from 'features/ui/layouts/auto-layout-context';
import type { PropsWithChildren } from 'react';
import { memo, useMemo, useRef } from 'react';

interface FocusRegionWrapperProps extends BoxProps {
  region: FocusRegionName;
  focusOnMount?: boolean;
}

export const BASE_FOCUS_REGION_STYLES: SystemStyleObject = {
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

export const FocusRegionWrapper = memo((props: FocusRegionWrapperProps) => {
  const { region, focusOnMount = false, sx: _sx, children, ...boxProps } = props;

  const ref = useRef<HTMLDivElement>(null);

  const sx = useMemo(() => ({ ...BASE_FOCUS_REGION_STYLES, ..._sx }), [_sx]);

  const shouldHighlightFocusedRegions = useAppSelector(selectSystemShouldEnableHighlightFocusedRegions);

  const options = useMemo(() => ({ focusOnMount }), [focusOnMount]);

  useFocusRegion(region, ref, options);
  const isFocused = useIsRegionFocused(region);
  const isHighlighted = isFocused && shouldHighlightFocusedRegions;

  return (
    <Box ref={ref} tabIndex={-1} sx={sx} data-highlighted={isHighlighted} {...boxProps}>
      {children}
    </Box>
  );
});

FocusRegionWrapper.displayName = 'FocusRegionWrapper';

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
      <Box ref={ref} tabIndex={-1} sx={BASE_FOCUS_REGION_STYLES} data-highlighted={isHighlighted}>
        {props.children}
      </Box>
    );
  }
);
AutoLayoutPanelContainer.displayName = 'AutoLayoutPanelContainer';

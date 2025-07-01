import { Box, type BoxProps, type SystemStyleObject } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { type FocusRegionName, useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { selectSystemShouldEnableHighlightFocusedRegions } from 'features/system/store/systemSlice';
import { memo, useMemo, useRef } from 'react';

interface FocusRegionWrapperProps extends BoxProps {
  region: FocusRegionName;
  focusOnMount?: boolean;
}

const BASE_FOCUS_REGION_STYLES: SystemStyleObject = {
  position: 'relative',
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
    // transition: 'border-color 0.1s ease-in-out',
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

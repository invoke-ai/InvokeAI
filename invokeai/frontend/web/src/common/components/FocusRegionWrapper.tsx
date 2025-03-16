import { Box, type BoxProps, type SystemStyleObject } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { type FocusRegionName, useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { selectSystemShouldEnableHighlightFocusedRegions } from 'features/system/store/systemSlice';
import { memo, useMemo, useRef } from 'react';

interface FocusRegionWrapperProps extends BoxProps {
  region: FocusRegionName;
  focusOnMount?: boolean;
}

const FOCUS_REGION_STYLES: SystemStyleObject = {
  position: 'relative',
  '&[data-highlighted="true"]::after': {
    borderColor: 'blue.700',
  },
  '&::after': {
    content: '""',
    position: 'absolute',
    inset: 0,
    zIndex: 1,
    borderRadius: 'base',
    border: '2px solid',
    borderColor: 'transparent',
    pointerEvents: 'none',
    transition: 'border-color 0.1s ease-in-out',
  },
};

export const FocusRegionWrapper = memo(
  ({ region, focusOnMount = false, sx, children, ...boxProps }: FocusRegionWrapperProps) => {
    const shouldHighlightFocusedRegions = useAppSelector(selectSystemShouldEnableHighlightFocusedRegions);

    const ref = useRef<HTMLDivElement>(null);

    const options = useMemo(() => ({ focusOnMount }), [focusOnMount]);

    useFocusRegion(region, ref, options);
    const isFocused = useIsRegionFocused(region);
    const isHighlighted = isFocused && shouldHighlightFocusedRegions;

    return (
      <Box
        ref={ref}
        tabIndex={-1}
        sx={useMemo(() => ({ ...FOCUS_REGION_STYLES, ...sx }), [sx])}
        data-highlighted={isHighlighted}
        {...boxProps}
      >
        {children}
      </Box>
    );
  }
);

FocusRegionWrapper.displayName = 'FocusRegionWrapper';

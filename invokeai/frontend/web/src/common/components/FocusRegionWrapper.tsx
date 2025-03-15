import { Box, type BoxProps, type SystemStyleObject } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { type FocusRegionName, useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { selectSystemShouldEnableHighlightFocusedRegions } from 'features/system/store/systemSlice';
import { forwardRef, memo, type RefObject, useMemo, useRef } from 'react';

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
  }
};

const FocusRegionWrapperComponent = forwardRef<HTMLDivElement, FocusRegionWrapperProps>(function RegionHighlighter(
  { region, focusOnMount = false, sx, children, ...boxProps },
  forwardedRef
) {
  const shouldHighlightFocusedRegions = useAppSelector(selectSystemShouldEnableHighlightFocusedRegions);

  const innerRef = useRef<HTMLDivElement>(null);
  const ref = (forwardedRef as RefObject<HTMLDivElement>) || innerRef;

  const options = useMemo(() => ({ focusOnMount }), [focusOnMount]);

  useFocusRegion(region, ref, options);
  const isFocused = useIsRegionFocused(region);
  const isHighlighted = isFocused && shouldHighlightFocusedRegions;

  return (
    <Box
      ref={ref}
      tabIndex={-1}
      sx={{ ...FOCUS_REGION_STYLES, ...sx}}
      data-highlighted={isHighlighted ? true : undefined}
      {...boxProps}
    >
      {children}
    </Box>
  );
});

FocusRegionWrapperComponent.displayName = 'FocusRegionWrapper';

export const FocusRegionWrapper = memo(FocusRegionWrapperComponent);

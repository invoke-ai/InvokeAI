import { Box, type BoxProps } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { type FocusRegionName, useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { selectSystemShouldEnableHighlightFocusedRegions } from 'features/system/store/systemSlice';
import { forwardRef, type RefObject, useMemo, useRef } from 'react';

interface FocusRegionWrapperProps extends BoxProps {
  region: FocusRegionName;
  focusOnMount?: boolean;
  highlightColor?: string;
}

export const FocusRegionWrapper = forwardRef<HTMLDivElement, FocusRegionWrapperProps>(function RegionHighlighter(
  { region, focusOnMount = false, highlightColor = 'blue.700', children, ...boxProps },
  forwardedRef
) {
  const shouldHighlightFocusedRegions = useAppSelector(selectSystemShouldEnableHighlightFocusedRegions);

  const innerRef = useRef<HTMLDivElement>(null);
  const ref = (forwardedRef as RefObject<HTMLDivElement>) || innerRef;

  const options = useMemo(() => ({ focusOnMount }), [focusOnMount]);

  useFocusRegion(region, ref, options);
  const isFocused = useIsRegionFocused(region);

  return (
    <Box
      ref={ref}
      position="relative"
      tabIndex={-1}
      _after={{
        content: '""',
        position: 'absolute',
        inset: 0,
        zIndex: 1,
        borderRadius: 'base',
        border: '2px solid',
        borderColor: isFocused && shouldHighlightFocusedRegions ? highlightColor : 'transparent',
        pointerEvents: 'none',
        transition: 'border-color 0.1s ease-in-out',
      }}
      {...boxProps}
    >
      {children}
    </Box>
  );
});

FocusRegionWrapper.displayName = 'FocusRegionWrapper';

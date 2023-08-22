import { useBreakpoint } from '@chakra-ui/react';

export default function useResolution():
  | 'mobile'
  | 'tablet'
  | 'desktop'
  | 'unknown' {
  const breakpointValue = useBreakpoint();

  const mobileResolutions = ['base', 'sm'];
  const tabletResolutions = ['md', 'lg'];
  const desktopResolutions = ['xl', '2xl'];

  if (mobileResolutions.includes(breakpointValue)) {
    return 'mobile';
  }
  if (tabletResolutions.includes(breakpointValue)) {
    return 'tablet';
  }
  if (desktopResolutions.includes(breakpointValue)) {
    return 'desktop';
  }
  return 'unknown';
}

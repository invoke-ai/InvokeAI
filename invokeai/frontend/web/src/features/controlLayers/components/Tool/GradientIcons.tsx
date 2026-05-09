import { Box } from '@invoke-ai/ui-library';
import { memo, useId } from 'react';

export const GradientToolIcon = memo(() => {
  const id = useId();
  const gradientId = `${id}-gradient-tool-horizontal`;
  return (
    <Box as="svg" viewBox="0 0 24 24" boxSize={6} aria-hidden focusable={false} display="block">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0.5" x2="1" y2="0.5">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.0" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0.85" />
        </linearGradient>
      </defs>
      <rect
        x="3"
        y="6"
        width="18"
        height="12"
        rx="2"
        fill={`url(#${gradientId})`}
        stroke="currentColor"
        strokeOpacity="0.9"
        strokeWidth="1"
      />
    </Box>
  );
});
GradientToolIcon.displayName = 'GradientToolIcon';

export const GradientLinearIcon = memo(() => {
  const id = useId();
  const gradientId = `${id}-gradient-linear-diagonal`;
  return (
    <Box as="svg" viewBox="0 0 24 24" boxSize="22px" aria-hidden focusable={false} display="block">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.0" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0.85" />
        </linearGradient>
      </defs>
      <rect
        x="4"
        y="4"
        width="16"
        height="16"
        rx="2"
        fill={`url(#${gradientId})`}
        stroke="currentColor"
        strokeOpacity="0.9"
        strokeWidth="1"
      />
    </Box>
  );
});
GradientLinearIcon.displayName = 'GradientLinearIcon';

export const GradientRadialIcon = memo(() => {
  const id = useId();
  const gradientId = `${id}-gradient-radial`;
  return (
    <Box as="svg" viewBox="0 0 24 24" boxSize="22px" aria-hidden focusable={false} display="block">
      <defs>
        <radialGradient id={gradientId} cx="0.5" cy="0.5" r="0.5">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.0" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0.85" />
        </radialGradient>
      </defs>
      <circle
        cx="12"
        cy="12"
        r="8"
        fill={`url(#${gradientId})`}
        stroke="currentColor"
        strokeOpacity="0.9"
        strokeWidth="1"
      />
    </Box>
  );
});
GradientRadialIcon.displayName = 'GradientRadialIcon';

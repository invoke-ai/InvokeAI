import type { LucideIcon } from 'lucide-react';

import { Box, Icon, ProgressCircle, Spinner } from '@chakra-ui/react';
import { CheckIcon, HourglassIcon } from 'lucide-react';

export type ProgressBadgeState = 'queued' | 'running' | 'completed';

export interface ProgressBadgeProps {
  state: ProgressBadgeState;
  /** Whole percent, or null while progress is unknown. */
  percentage: number | null;
  /** Names the indicator; omit when the parent already names the state, which hides the badge from assistive tech. */
  label?: string;
  /** Replaces the state indicator, e.g. a pinned marker. */
  icon?: LucideIcon;
}

// Matches the ring's `xs` size (14px, 2px stroke) so every state occupies the same footprint.
const INDICATOR_SIZE = '3.5';

/** Corner badge on a thumbnail showing a job's progress state: queued, running (ring or spinner), or completed. */
export const ProgressBadge = ({ state, percentage, label, icon }: ProgressBadgeProps) => (
  <Box
    aria-hidden={label === undefined || undefined}
    bg="bg/85"
    bottom="1"
    display="flex"
    p="0.5"
    pointerEvents="none"
    position="absolute"
    right="1"
    rounded="full"
    zIndex="1"
  >
    {icon || state !== 'running' ? (
      <Icon
        aria-label={label}
        as={icon ?? (state === 'queued' ? HourglassIcon : CheckIcon)}
        boxSize={INDICATOR_SIZE}
        role={label === undefined ? undefined : 'img'}
      />
    ) : percentage === null ? (
      // The app-wide spinner, not the ring's indeterminate sweep, for unknown progress.
      <Spinner aria-label={label} borderWidth="2px" boxSize={INDICATOR_SIZE} role="progressbar" />
    ) : (
      <ProgressCircle.Root aria-label={label} size="xs" value={percentage}>
        <ProgressCircle.Circle>
          <ProgressCircle.Track />
          <ProgressCircle.Range />
        </ProgressCircle.Circle>
      </ProgressCircle.Root>
    )}
  </Box>
);

import type { CircularProgressProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { CircularProgress, Text, Tooltip } from '@invoke-ai/ui-library';
import { useProgressDeviceLabel } from 'common/hooks/useProgressDeviceLabel';
import type { ComponentRef } from 'react';
import { forwardRef, memo } from 'react';
import type { S } from 'services/api/types';
import { formatProgressMessage } from 'services/events/stores';

const circleStyles: SystemStyleObject = {
  // The callers position this circle with `position="absolute"`, which makes it the containing
  // block for the absolutely-centered GPU label below. Do NOT set `position` here — an `sx` value
  // would override the caller's prop and break the circle's corner anchoring.
  circle: {
    transitionProperty: 'none',
    transitionDuration: '0s',
  },
};

// Centered GPU-number label drawn inside the ring (CircularProgressLabel isn't exported by the ui-library).
const labelStyles: SystemStyleObject = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  fontSize: '0.6rem',
  lineHeight: 1,
  fontWeight: 'bold',
  color: 'invokeBlue.300',
  textShadow: '0 0 3px var(--invoke-colors-base-900)',
  pointerEvents: 'none',
};

type ProgressDeviceLabel = ReturnType<typeof useProgressDeviceLabel>;

// The circle is split out and memoized so it does NOT re-render when only the tooltip message
// changes. Every progress event re-renders the parent, and during the indeterminate phases
// (everything except denoising) those events keep the same `isIndeterminate`/`value` — but
// re-rendering the CircularProgress restarts its CSS spin animation, which reads as the disk
// "flashing". Memoizing on the visual props keeps the animation continuous. forwardRef so the
// wrapping Tooltip can still anchor to it.
const ProgressCircle = memo(
  forwardRef<ComponentRef<typeof CircularProgress>, { deviceLabel: ProgressDeviceLabel } & CircularProgressProps>(
    ({ deviceLabel, ...rest }, ref) => (
      <CircularProgress ref={ref} size="14px" color="invokeBlue.500" thickness={14} sx={circleStyles} {...rest}>
        {deviceLabel && <Text sx={labelStyles}>{deviceLabel.index}</Text>}
      </CircularProgress>
    )
  )
);
ProgressCircle.displayName = 'ProgressCircle';

export const ProgressIndicator = memo(
  ({ progressEvent, ...rest }: { progressEvent: S['InvocationProgressEvent'] } & CircularProgressProps) => {
    const deviceLabel = useProgressDeviceLabel(progressEvent?.device);
    const message = formatProgressMessage(progressEvent);
    return (
      <Tooltip label={deviceLabel ? `${deviceLabel.name} — ${message}` : message}>
        <ProgressCircle
          isIndeterminate={!progressEvent || progressEvent.percentage === null}
          value={progressEvent?.percentage ? progressEvent.percentage * 100 : undefined}
          deviceLabel={deviceLabel}
          {...rest}
        />
      </Tooltip>
    );
  }
);
ProgressIndicator.displayName = 'ProgressMessage';

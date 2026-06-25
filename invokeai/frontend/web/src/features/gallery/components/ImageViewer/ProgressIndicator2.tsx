import type { CircularProgressProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { CircularProgress, Text, Tooltip } from '@invoke-ai/ui-library';
import { useProgressDeviceLabel } from 'common/hooks/useProgressDeviceLabel';
import { memo } from 'react';
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

export const ProgressIndicator = memo(
  ({ progressEvent, ...rest }: { progressEvent: S['InvocationProgressEvent'] } & CircularProgressProps) => {
    const deviceLabel = useProgressDeviceLabel(progressEvent?.device);
    const message = formatProgressMessage(progressEvent);
    return (
      <Tooltip label={deviceLabel ? `${deviceLabel.name} — ${message}` : message}>
        <CircularProgress
          size="14px"
          color="invokeBlue.500"
          thickness={14}
          isIndeterminate={!progressEvent || progressEvent.percentage === null}
          value={progressEvent?.percentage ? progressEvent.percentage * 100 : undefined}
          sx={circleStyles}
          {...rest}
        >
          {deviceLabel && <Text sx={labelStyles}>{deviceLabel.index}</Text>}
        </CircularProgress>
      </Tooltip>
    );
  }
);
ProgressIndicator.displayName = 'ProgressMessage';

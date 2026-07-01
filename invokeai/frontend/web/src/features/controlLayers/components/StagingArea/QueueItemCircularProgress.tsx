import type { CircularProgressProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { CircularProgress, Text, Tooltip } from '@invoke-ai/ui-library';
import { useProgressDeviceLabel } from 'common/hooks/useProgressDeviceLabel';
import { getProgressMessage } from 'features/controlLayers/components/StagingArea/shared';
import { memo } from 'react';
import type { S } from 'services/api/types';

import { useProgressDatum } from './context';

const circleStyles: SystemStyleObject = {
  circle: {
    transitionProperty: 'none',
    transitionDuration: '0s',
  },
  position: 'absolute',
  top: 2,
  right: 2,
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

type Props = { itemId: number; status: S['SessionQueueItem']['status'] } & CircularProgressProps;

export const QueueItemCircularProgress = memo(({ itemId, status, ...rest }: Props) => {
  const { progressEvent } = useProgressDatum(itemId);
  const deviceLabel = useProgressDeviceLabel(progressEvent?.device);

  if (status !== 'in_progress') {
    return null;
  }

  const message = getProgressMessage(progressEvent);

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
});
QueueItemCircularProgress.displayName = 'QueueItemCircularProgress';

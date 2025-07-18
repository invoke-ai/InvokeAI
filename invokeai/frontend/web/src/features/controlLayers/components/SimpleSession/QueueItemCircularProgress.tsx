import type { CircularProgressProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { CircularProgress, Tooltip } from '@invoke-ai/ui-library';
import { getProgressMessage } from 'features/controlLayers/components/SimpleSession/shared';
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

type Props = { itemId: number; status: S['SessionQueueItem']['status'] } & CircularProgressProps;

export const QueueItemCircularProgress = memo(({ itemId, status, ...rest }: Props) => {
  const { progressEvent } = useProgressDatum(itemId);

  if (status !== 'in_progress') {
    return null;
  }

  return (
    <Tooltip label={getProgressMessage(progressEvent)}>
      <CircularProgress
        size="14px"
        color="invokeBlue.500"
        thickness={14}
        isIndeterminate={!progressEvent || progressEvent.percentage === null}
        value={progressEvent?.percentage ? progressEvent.percentage * 100 : undefined}
        sx={circleStyles}
        {...rest}
      />
    </Tooltip>
  );
});
QueueItemCircularProgress.displayName = 'QueueItemCircularProgress';

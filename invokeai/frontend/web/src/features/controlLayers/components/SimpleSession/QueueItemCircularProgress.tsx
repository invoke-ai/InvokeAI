import type { CircularProgressProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { CircularProgress, Tooltip } from '@invoke-ai/ui-library';
import { useCanvasSessionContext,useProgressData } from 'features/controlLayers/components/SimpleSession/context';
import { getProgressMessage } from 'features/controlLayers/components/SimpleSession/shared';
import { memo } from 'react';
import type { S } from 'services/api/types';

const circleStyles: SystemStyleObject = {
  circle: {
    transitionProperty: 'none',
    transitionDuration: '0s',
  },
  position: 'absolute',
  top: 2,
  right: 2,
};

export const QueueItemCircularProgress = memo(
  ({
    session_id,
    status,
    ...rest
  }: { session_id: string; status: S['SessionQueueItem']['status'] } & CircularProgressProps) => {
    const { $progressData } = useCanvasSessionContext();
    const { progressEvent } = useProgressData($progressData, session_id);

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
  }
);
QueueItemCircularProgress.displayName = 'QueueItemCircularProgress';

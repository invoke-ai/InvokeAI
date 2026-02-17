import type { CircularProgressProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { CircularProgress } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import { memo } from 'react';
import type { S } from 'services/api/types';
import { formatProgressMessage } from 'services/events/stores';

const circleStyles: SystemStyleObject = {
  circle: {
    transitionProperty: 'none',
    transitionDuration: '0s',
  },
};

export const ProgressIndicator = memo(
  ({ progressEvent, ...rest }: { progressEvent: S['InvocationProgressEvent'] } & CircularProgressProps) => {
    return (
      <IAITooltip label={formatProgressMessage(progressEvent)}>
        <CircularProgress
          size="14px"
          color="invokeBlue.500"
          thickness={14}
          isIndeterminate={!progressEvent || progressEvent.percentage === null}
          value={progressEvent?.percentage ? progressEvent.percentage * 100 : undefined}
          sx={circleStyles}
          {...rest}
        />
      </IAITooltip>
    );
  }
);
ProgressIndicator.displayName = 'ProgressMessage';

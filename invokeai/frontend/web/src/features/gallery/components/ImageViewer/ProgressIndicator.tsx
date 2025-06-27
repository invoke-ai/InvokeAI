import type { CircularProgressProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { CircularProgress, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { memo } from 'react';
import { useIsNonPromptExpansionGenerationInProgress } from 'services/api/endpoints/queue';
import { $lastProgressEvent, formatProgressMessage } from 'services/events/stores';

const circleStyles: SystemStyleObject = {
  circle: {
    transitionProperty: 'none',
    transitionDuration: '0s',
  },
};

export const ProgressIndicator = memo((props: CircularProgressProps) => {
  const isNonPromptExpansionGenerationInProgress = useIsNonPromptExpansionGenerationInProgress();
  const lastProgressEvent = useStore($lastProgressEvent);
  if (!isNonPromptExpansionGenerationInProgress) {
    return null;
  }
  if (!lastProgressEvent) {
    return null;
  }
  return (
    <Tooltip label={formatProgressMessage(lastProgressEvent)}>
      <CircularProgress
        size="14px"
        color="invokeBlue.500"
        thickness={14}
        isIndeterminate={!lastProgressEvent || lastProgressEvent.percentage === null}
        value={lastProgressEvent?.percentage ? lastProgressEvent.percentage * 100 : undefined}
        sx={circleStyles}
        {...props}
      />
    </Tooltip>
  );
});
ProgressIndicator.displayName = 'ProgressMessage';

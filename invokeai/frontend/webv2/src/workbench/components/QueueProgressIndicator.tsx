import type { QueueProgressBarState } from '@features/queue/contracts';
import type { ComponentProps } from 'react';

import { ProgressCircle } from '@chakra-ui/react';

const getProgressValuePercent = (state: QueueProgressBarState): number | null =>
  state.kind === 'determinate' ? state.value * 100 : state.value;

type ProgressCircleRootProps = ComponentProps<typeof ProgressCircle.Root>;
type QueueCircularProgressSize = NonNullable<ProgressCircleRootProps['size']> | '2xs';

export const QueueCircularProgress = ({
  size = '2xs',
  state,
  ...props
}: Omit<ProgressCircleRootProps, 'size' | 'value'> & {
  size?: QueueCircularProgressSize;
  state: QueueProgressBarState;
}) => {
  if (state.kind === 'idle') {
    return null;
  }

  return (
    <ProgressCircle.Root
      aria-label="Project queue progress"
      colorPalette="accent"
      flexShrink="0"
      size={size as ProgressCircleRootProps['size']}
      value={getProgressValuePercent(state)}
      {...props}
    >
      <ProgressCircle.Circle>
        <ProgressCircle.Track stroke="{colors.border.subtle}" />
        <ProgressCircle.Range stroke="{colors.accent.solid}" />
      </ProgressCircle.Circle>
    </ProgressCircle.Root>
  );
};

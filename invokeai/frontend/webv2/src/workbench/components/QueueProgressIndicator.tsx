import type { QueueProgressBarState } from '@features/queue/contracts';
import type { ComponentProps } from 'react';

import { Progress, ProgressCircle } from '@chakra-ui/react';

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

export const QueueTabBackgroundProgress = ({
  state,
  ...props
}: Omit<ComponentProps<typeof Progress.Root>, 'value'> & { state: QueueProgressBarState }) => {
  if (state.kind === 'idle') {
    return null;
  }

  return (
    <Progress.Root
      aria-label="Widget queue progress"
      h="full"
      inset="0"
      max={1}
      pointerEvents="none"
      position="absolute"
      value={state.value}
      {...props}
    >
      <Progress.Track bg="transparent" h="full" boxShadow="none">
        <Progress.Range bg="accent.subtle" />
      </Progress.Track>
    </Progress.Root>
  );
};

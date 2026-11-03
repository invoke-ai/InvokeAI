import type { QueueProgressBarState } from '@features/queue/contracts';
import type { ComponentProps } from 'react';

import { ProgressCircle } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

const getProgressValuePercent = (state: QueueProgressBarState): number | null =>
  state.kind === 'determinate' ? state.value * 100 : state.value;

type ProgressCircleRootProps = ComponentProps<typeof ProgressCircle.Root>;

export const QueueCircularProgress = ({
  size = 'sm',
  state,
  ...props
}: Omit<ProgressCircleRootProps, 'value'> & {
  state: QueueProgressBarState;
}) => {
  const { t } = useTranslation();

  if (state.kind === 'idle') {
    return null;
  }

  return (
    <ProgressCircle.Root
      aria-label={t('topbar.queue.projectProgress')}
      colorPalette="accent"
      flexShrink="0"
      size={size}
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

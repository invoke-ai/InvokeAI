import type { TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { useCanvasSessionContext, useProgressData } from 'features/controlLayers/components/SimpleSession/context';
import { memo } from 'react';
import type { S } from 'services/api/types';

type Props = { item: S['SessionQueueItem'] } & TextProps;

export const QueueItemStatusLabel = memo(({ item, ...rest }: Props) => {
  const ctx = useCanvasSessionContext();
  const { progressImage, imageLoaded } = useProgressData(ctx.$progressData, item.item_id);

  if (progressImage || imageLoaded) {
    return null;
  }

  if (item.status === 'pending') {
    return (
      <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="base.300" {...rest}>
        Pending
      </Text>
    );
  }
  if (item.status === 'canceled') {
    return (
      <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="warning.300" {...rest}>
        Canceled
      </Text>
    );
  }
  if (item.status === 'failed') {
    return (
      <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="error.300" {...rest}>
        Failed
      </Text>
    );
  }

  if (item.status === 'in_progress') {
    return (
      <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="invokeBlue.300" {...rest}>
        In Progress
      </Text>
    );
  }

  if (item.status === 'completed') {
    return (
      <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="invokeGreen.300" {...rest}>
        Completed
      </Text>
    );
  }

  return null;
});
QueueItemStatusLabel.displayName = 'QueueItemStatusLabel';

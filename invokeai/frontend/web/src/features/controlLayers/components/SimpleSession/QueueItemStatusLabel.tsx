import type { TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { S } from 'services/api/types';

import { useProgressDatum } from './context2';

type Props = { item: S['SessionQueueItem'] } & TextProps;

export const QueueItemStatusLabel = memo(({ item, ...rest }: Props) => {
  const { progressImage } = useProgressDatum(item.item_id);

  if (progressImage) {
    return null;
  }

  if (item.status === 'pending') {
    return (
      <Text fontSize="xs" pointerEvents="none" userSelect="none" fontWeight="semibold" color="base.300" {...rest}>
        Pending
      </Text>
    );
  }
  if (item.status === 'canceled') {
    return (
      <Text fontSize="xs" pointerEvents="none" userSelect="none" fontWeight="semibold" color="warning.300" {...rest}>
        Canceled
      </Text>
    );
  }
  if (item.status === 'failed') {
    return (
      <Text fontSize="xs" pointerEvents="none" userSelect="none" fontWeight="semibold" color="error.300" {...rest}>
        Failed
      </Text>
    );
  }

  if (item.status === 'in_progress') {
    return (
      <Text fontSize="xs" pointerEvents="none" userSelect="none" fontWeight="semibold" color="invokeBlue.300" {...rest}>
        In Progress
      </Text>
    );
  }

  if (item.status === 'completed') {
    return (
      <Text
        fontSize="xs"
        pointerEvents="none"
        userSelect="none"
        fontWeight="semibold"
        color="invokeGreen.300"
        {...rest}
      >
        Completed
      </Text>
    );
  }

  return null;
});
QueueItemStatusLabel.displayName = 'QueueItemStatusLabel';

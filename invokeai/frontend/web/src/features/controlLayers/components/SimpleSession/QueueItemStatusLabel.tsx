/* eslint-disable i18next/no-literal-string */
import type { TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { S } from 'services/api/types';

export const QueueItemStatusLabel = memo(
  ({ status, ...rest }: { status: S['SessionQueueItem']['status'] } & TextProps) => {
    if (status === 'pending') {
      return (
        <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="base.300" {...rest}>
          Pending
        </Text>
      );
    }
    if (status === 'canceled') {
      return (
        <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="warning.300" {...rest}>
          Canceled
        </Text>
      );
    }
    if (status === 'failed') {
      return (
        <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="error.300" {...rest}>
          Failed
        </Text>
      );
    }

    if (status === 'in_progress') {
      return (
        <Text pointerEvents="none" userSelect="none" fontWeight="semibold" color="invokeBlue.300" {...rest}>
          In Progress
        </Text>
      );
    }

    return null;
  }
);
QueueItemStatusLabel.displayName = 'QueueItemStatusLabel';

/* eslint-disable i18next/no-literal-string */
import type { TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { DROP_SHADOW, getProgressMessage } from 'features/controlLayers/components/SimpleSession/shared';
import { memo } from 'react';
import type { S } from 'services/api/types';
import { useProgressData } from 'services/events/stores';

export const QueueItemProgressMessage = memo(
  ({ session_id, status, ...rest }: { session_id: string; status: S['SessionQueueItem']['status'] } & TextProps) => {
    const { $progressData } = useCanvasSessionContext();
    const { progressEvent } = useProgressData($progressData, session_id);

    if (status === 'completed' || status === 'failed' || status === 'canceled') {
      return null;
    }

    if (status === 'pending') {
      return (
        <Text pointerEvents="none" userSelect="none" filter={DROP_SHADOW} {...rest}>
          Waiting to start...
        </Text>
      );
    }

    return (
      <Text pointerEvents="none" userSelect="none" filter={DROP_SHADOW} {...rest}>
        {getProgressMessage(progressEvent)}
      </Text>
    );
  }
);
QueueItemProgressMessage.displayName = 'QueueItemProgressMessage';

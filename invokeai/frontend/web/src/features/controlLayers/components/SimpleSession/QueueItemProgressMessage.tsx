/* eslint-disable i18next/no-literal-string */
import type { TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { useCanvasSessionContext, useProgressData } from 'features/controlLayers/components/SimpleSession/context';
import { DROP_SHADOW, getProgressMessage } from 'features/controlLayers/components/SimpleSession/shared';
import { memo } from 'react';
import type { S } from 'services/api/types';

type Props = { itemId: number; status: S['SessionQueueItem']['status'] } & TextProps;

export const QueueItemProgressMessage = memo(({ itemId, status, ...rest }: Props) => {
  const ctx = useCanvasSessionContext();
  const { progressEvent } = useProgressData(ctx.$progressData, itemId);

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
});
QueueItemProgressMessage.displayName = 'QueueItemProgressMessage';

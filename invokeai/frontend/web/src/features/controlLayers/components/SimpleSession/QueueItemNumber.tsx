import type { TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { DROP_SHADOW } from 'features/controlLayers/components/SimpleSession/shared';
import { memo } from 'react';

export const QueueItemNumber = memo(({ number, ...rest }: { number: number } & TextProps) => {
  return <Text pointerEvents="none" userSelect="none" filter={DROP_SHADOW} {...rest}>{`#${number}`}</Text>;
});
QueueItemNumber.displayName = 'QueueItemNumber';

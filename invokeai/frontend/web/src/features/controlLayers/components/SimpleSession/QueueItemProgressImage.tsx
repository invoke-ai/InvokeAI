import type { ImageProps } from '@invoke-ai/ui-library';
import { Image } from '@invoke-ai/ui-library';
import { useCanvasSessionContext, useProgressData } from 'features/controlLayers/components/SimpleSession/context';
import { memo } from 'react';

type Props = { itemId: number } & ImageProps;

export const QueueItemProgressImage = memo(({ itemId, ...rest }: Props) => {
  const ctx = useCanvasSessionContext();
  const { progressImage } = useProgressData(ctx.$progressData, itemId);

  if (!progressImage) {
    return null;
  }

  return (
    <Image
      objectFit="contain"
      maxH="full"
      maxW="full"
      draggable={false}
      src={progressImage.dataURL}
      width={progressImage.width}
      height={progressImage.height}
      {...rest}
    />
  );
});
QueueItemProgressImage.displayName = 'QueueItemProgressImage';

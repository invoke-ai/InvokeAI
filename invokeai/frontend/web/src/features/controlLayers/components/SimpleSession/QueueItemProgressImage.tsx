import type { ImageProps } from '@invoke-ai/ui-library';
import { Image } from '@invoke-ai/ui-library';
import { memo } from 'react';

import { useProgressDatum } from './context2';

type Props = { itemId: number } & ImageProps;

export const QueueItemProgressImage = memo(({ itemId, ...rest }: Props) => {
  const { progressImage } = useProgressDatum(itemId);

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

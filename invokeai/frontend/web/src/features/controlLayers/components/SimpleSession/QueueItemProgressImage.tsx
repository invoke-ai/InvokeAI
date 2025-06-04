import type { ImageProps } from '@invoke-ai/ui-library';
import { Image } from '@invoke-ai/ui-library';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { memo } from 'react';
import { useProgressData } from 'services/events/stores';

export const QueueItemProgressImage = memo(({ session_id, ...rest }: { session_id: string } & ImageProps) => {
  const { $progressData } = useCanvasSessionContext();
  const { progressImage } = useProgressData($progressData, session_id);

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

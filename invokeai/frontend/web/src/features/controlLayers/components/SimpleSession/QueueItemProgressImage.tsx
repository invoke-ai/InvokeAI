import type { ImageProps } from '@invoke-ai/ui-library';
import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { useCanvasSessionContext, useProgressData } from 'features/controlLayers/components/SimpleSession/context';
import { memo } from 'react';
import { PiImageBold } from 'react-icons/pi';

type Props = { itemId: number } & ImageProps;

export const QueueItemProgressImage = memo(({ itemId, ...rest }: Props) => {
  const ctx = useCanvasSessionContext();
  const { progressImage } = useProgressData(ctx.$progressData, itemId);

  if (!progressImage) {
    return (
      <Flex w="full" h="full" bg="base.700" alignItems="center" justifyContent="center">
        <Icon as={PiImageBold} boxSize={16} opacity={0.2} />
      </Flex>
    );
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

import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { ImageDTO, VideoDTO } from 'services/api/types';

type Props = {
  itemDTO: ImageDTO | VideoDTO;
};

export const GalleryItemSizeBadge = memo(({ itemDTO }: Props) => {
  return (
    <Text
      className="gallery-image-size-badge"
      position="absolute"
      background="base.900"
      color="base.50"
      fontSize="sm"
      fontWeight="semibold"
      bottom={1}
      left={1}
      opacity={0.7}
      px={2}
      lineHeight={1.25}
      borderTopEndRadius="base"
      pointerEvents="none"
    >{`${itemDTO.width}x${itemDTO.height}`}</Text>
  );
});

GalleryItemSizeBadge.displayName = 'GalleryItemSizeBadge';

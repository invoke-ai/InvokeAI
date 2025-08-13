import { Badge } from '@invoke-ai/ui-library';
import { isVideoResource } from 'features/gallery/store/resourceTypes';
import { memo } from 'react';
import { ImageDTO, VideoDTO } from 'services/api/types';

type Props = {
  resource: ImageDTO | VideoDTO;
  shouldShow: boolean;
};

export const GalleryResourceSizeBadge = memo(({ resource, shouldShow }: Props) => {
  if (!shouldShow) {
    return null;
  }

  const getBadgeText = () => {
    if (isVideoResource(resource)) {
      return `${resource.width}×${resource.height}`;
    } else {
      return `${resource.width}×${resource.height}`;
    }
  };

  return (
    <Badge
      className="gallery-resource-size-badge"
      position="absolute"
      bottom={1}
      insetInlineStart={1}
      colorScheme="base"
      variant="solid"
      fontSize={10}
      size="xs"
      userSelect="none"
      opacity={0.7}
    >
      {getBadgeText()}
    </Badge>
  );
});

GalleryResourceSizeBadge.displayName = 'GalleryResourceSizeBadge';


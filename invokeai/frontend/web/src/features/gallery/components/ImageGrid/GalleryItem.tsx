import { GalleryImage } from 'features/gallery/components/ImageGrid/GalleryImage';
import { GalleryResource } from 'features/gallery/components/ImageGrid/GalleryResource';
import { memo } from 'react';
import type { ImageDTO, VideoDTO } from 'services/api/types';

interface GalleryItemProps {
  item: ImageDTO | VideoDTO;
}

/**
 * A wrapper component that can render either legacy ImageDTO items using GalleryImage
 * or new GalleryResource items using GalleryResource.
 */
export const GalleryItem = memo(({ item }: GalleryItemProps) => {
  // Check if the item is an ImageDTO (legacy) or GalleryResource (new)
  if ('image_name' in item) {
    // This is a legacy ImageDTO, use the old GalleryImage component
    return <GalleryImage imageDTO={item} />;
  } else {
    // This is a GalleryResource, use the new GalleryResource component
    return <GalleryResource resource={item} />;
  }
});

GalleryItem.displayName = 'GalleryItem';



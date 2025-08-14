import { GalleryImage } from 'features/gallery/components/ImageGrid/GalleryImage';
import { memo } from 'react';
import type { ImageDTO, VideoDTO } from 'services/api/types';

import { GalleryVideo } from './GalleryVideo';

interface GalleryItemProps {
  item: ImageDTO | VideoDTO;
}

export const GalleryItem = memo(({ item }: GalleryItemProps) => {
  if ('image_name' in item) {
    return <GalleryImage imageDTO={item} />;
  } else {
    return <GalleryVideo videoDTO={item} />;
  }
});

GalleryItem.displayName = 'GalleryItem';



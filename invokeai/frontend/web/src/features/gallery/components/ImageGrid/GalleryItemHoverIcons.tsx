import { useAppSelector } from 'app/store/storeHooks';
import { GalleryItemDeleteIconButton } from 'features/gallery/components/ImageGrid/GalleryItemDeleteIconButton';
import { GalleryItemOpenInViewerIconButton } from 'features/gallery/components/ImageGrid/GalleryItemOpenInViewerIconButton';
import { GalleryItemSizeBadge } from 'features/gallery/components/ImageGrid/GalleryItemSizeBadge';
import { GalleryItemStarIconButton } from 'features/gallery/components/ImageGrid/GalleryItemStarIconButton';
import { selectAlwaysShouldImageSizeBadge } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import type { ImageDTO, VideoDTO } from 'services/api/types';

type Props = {
  itemDTO: ImageDTO | VideoDTO;
  isHovered: boolean;
};

export const GalleryItemHoverIcons = memo(({ itemDTO, isHovered }: Props) => {
  const alwaysShowImageSizeBadge = useAppSelector(selectAlwaysShouldImageSizeBadge);

  return (
    <>
      {(isHovered || alwaysShowImageSizeBadge) && <GalleryItemSizeBadge itemDTO={itemDTO} />}
      {(isHovered || itemDTO.starred) && <GalleryItemStarIconButton itemDTO={itemDTO} />}
      {isHovered && <GalleryItemDeleteIconButton itemDTO={itemDTO} />}
      {isHovered && <GalleryItemOpenInViewerIconButton itemDTO={itemDTO} />}
    </>
  );
});

GalleryItemHoverIcons.displayName = 'GalleryItemHoverIcons';

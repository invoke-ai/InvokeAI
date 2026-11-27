import { useAppSelector } from 'app/store/storeHooks';
import { GalleryItemDeleteIconButton } from 'features/gallery/components/ImageGrid/GalleryItemDeleteIconButton';
import { GalleryItemOpenInViewerIconButton } from 'features/gallery/components/ImageGrid/GalleryItemOpenInViewerIconButton';
import { GalleryItemSizeBadge } from 'features/gallery/components/ImageGrid/GalleryItemSizeBadge';
import { GalleryItemStarIconButton } from 'features/gallery/components/ImageGrid/GalleryItemStarIconButton';
import { selectAlwaysShouldImageSizeBadge } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import type { ImageDTO } from 'services/api/types';

type Props = {
  imageDTO: ImageDTO;
  isHovered: boolean;
};

export const GalleryItemHoverIcons = memo(({ imageDTO, isHovered }: Props) => {
  const alwaysShowImageSizeBadge = useAppSelector(selectAlwaysShouldImageSizeBadge);

  return (
    <>
      {(isHovered || alwaysShowImageSizeBadge) && <GalleryItemSizeBadge imageDTO={imageDTO} />}
      {(isHovered || imageDTO.starred) && <GalleryItemStarIconButton imageDTO={imageDTO} />}
      {isHovered && <GalleryItemDeleteIconButton imageDTO={imageDTO} />}
      {isHovered && <GalleryItemOpenInViewerIconButton imageDTO={imageDTO} />}
    </>
  );
});

GalleryItemHoverIcons.displayName = 'GalleryItemHoverIcons';

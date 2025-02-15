import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { GalleryImageDeleteIconButton } from 'features/gallery/components/ImageGrid/GalleryImageDeleteIconButton';
import { GalleryImageOpenInViewerIconButton } from 'features/gallery/components/ImageGrid/GalleryImageOpenInViewerIconButton';
import { GalleryImageSizeBadge } from 'features/gallery/components/ImageGrid/GalleryImageSizeBadge';
import { GalleryImageStarIconButton } from 'features/gallery/components/ImageGrid/GalleryImageStarIconButton';
import { selectAlwaysShouldImageSizeBadge } from 'features/gallery/store/gallerySelectors';
import type { Atom } from 'nanostores';
import { memo } from 'react';
import type { ImageDTO } from 'services/api/types';

type Props = {
  imageDTO: ImageDTO;
  $isHovered: Atom<boolean>;
};

export const GalleryImageHoverIcons = memo(({ imageDTO, $isHovered }: Props) => {
  const alwaysShowImageSizeBadge = useAppSelector(selectAlwaysShouldImageSizeBadge);
  const isHovered = useStore($isHovered);

  return (
    <>
      {(isHovered || alwaysShowImageSizeBadge) && <GalleryImageSizeBadge imageDTO={imageDTO} />}
      {(isHovered || imageDTO.starred) && <GalleryImageStarIconButton imageDTO={imageDTO} />}
      {isHovered && <GalleryImageDeleteIconButton imageDTO={imageDTO} />}
      {isHovered && <GalleryImageOpenInViewerIconButton imageDTO={imageDTO} />}
    </>
  );
});

GalleryImageHoverIcons.displayName = 'GalleryImageHoverIcons';

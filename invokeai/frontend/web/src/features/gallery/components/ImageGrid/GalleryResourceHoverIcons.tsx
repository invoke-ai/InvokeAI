import { useAppSelector } from 'app/store/storeHooks';
import { GalleryResourceDeleteIconButton } from 'features/gallery/components/ImageGrid/GalleryResourceDeleteIconButton';
import { GalleryResourceOpenInViewerIconButton } from 'features/gallery/components/ImageGrid/GalleryResourceOpenInViewerIconButton';
import { GalleryResourceSizeBadge } from 'features/gallery/components/ImageGrid/GalleryResourceSizeBadge';
import { GalleryResourceStarIconButton } from 'features/gallery/components/ImageGrid/GalleryResourceStarIconButton';
import { selectAlwaysShouldImageSizeBadge } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { ImageDTO, VideoDTO } from 'services/api/types';

type Props = {
  resource: ImageDTO | VideoDTO;
  isHovered: boolean;
};

export const GalleryResourceHoverIcons = memo(({ resource, isHovered }: Props) => {
  const shouldShowImageSizeBadge = useAppSelector(selectAlwaysShouldImageSizeBadge);

  return (
    <>
      <GalleryResourceDeleteIconButton resource={resource} isHovered={isHovered} />
      <GalleryResourceStarIconButton resource={resource} isHovered={isHovered} />
      <GalleryResourceOpenInViewerIconButton resource={resource} isHovered={isHovered} />
      <GalleryResourceSizeBadge resource={resource} shouldShow={shouldShowImageSizeBadge || isHovered} />
    </>
  );
});

GalleryResourceHoverIcons.displayName = 'GalleryResourceHoverIcons';


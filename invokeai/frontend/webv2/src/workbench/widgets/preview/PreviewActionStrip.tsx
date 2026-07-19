import type { GalleryImage } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { HStack, Icon } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { CopyIcon, DownloadIcon, EllipsisVerticalIcon, ImagesIcon, StarIcon, type LucideIcon } from 'lucide-react';
import { useCallback, type MouseEvent } from 'react';

import type { PreviewDensity } from './previewDensity';

/**
 * The header's always-visible action row: quick verbs one click away, plus an
 * "image actions" dropdown that opens the full right-click context menu
 * (anchored under the button) — one source of truth for every image verb,
 * mirroring the legacy viewer's menu button. Compact densities collapse to
 * star + the dropdown.
 */
export const PreviewActionStrip = ({
  actions,
  density,
  image,
  onOpenMenu,
}: {
  actions: ImageActions;
  density: PreviewDensity;
  image: GalleryImage;
  /** Opens the view's image context menu at viewport coordinates. */
  onOpenMenu: ((x: number, y: number) => void) | null;
}) => {
  const toggleStar = useCallback(
    () => void actions.setImagesStarred([image.imageName], !image.starred),
    [actions, image.imageName, image.starred]
  );
  const selectForCompare = useCallback(() => actions.selectForCompare(image), [actions, image]);
  const copyImage = useCallback(() => void actions.copyImage(image), [actions, image]);
  const downloadImage = useCallback(() => void actions.downloadImage(image), [actions, image]);
  const openMenu = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();

      onOpenMenu?.(rect.left, rect.bottom + 4);
    },
    [onOpenMenu]
  );
  const starLabel = image.starred ? 'Unstar image' : 'Star image';
  const starButton = (
    <Tooltip content={starLabel}>
      <IconButton aria-label={starLabel} color="fg.muted" size="2xs" variant="ghost" onClick={toggleStar}>
        <Icon as={StarIcon} boxSize="3.5" fill={image.starred ? 'currentColor' : 'none'} />
      </IconButton>
    </Tooltip>
  );
  const menuButton = onOpenMenu ? (
    <Tooltip content="Image actions">
      <IconButton aria-label="Image actions" color="fg.muted" size="2xs" variant="ghost" onClick={openMenu}>
        <Icon as={EllipsisVerticalIcon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  ) : null;

  if (density !== 'full') {
    return (
      <HStack flexShrink={0} gap="0.5">
        {starButton}
        {menuButton}
      </HStack>
    );
  }

  return (
    <HStack flexShrink={0} gap="0.5">
      {starButton}
      <StripIconButton icon={ImagesIcon} label="Select for Compare" onClick={selectForCompare} />
      <StripIconButton icon={CopyIcon} label="Copy to clipboard" onClick={copyImage} />
      <StripIconButton icon={DownloadIcon} label="Download image" onClick={downloadImage} />
      {menuButton}
    </HStack>
  );
};

const StripIconButton = ({ icon, label, onClick }: { icon: LucideIcon; label: string; onClick: () => void }) => (
  <Tooltip content={label}>
    <IconButton aria-label={label} color="fg.muted" size="2xs" variant="ghost" onClick={onClick}>
      <Icon as={icon} boxSize="3.5" />
    </IconButton>
  </Tooltip>
);

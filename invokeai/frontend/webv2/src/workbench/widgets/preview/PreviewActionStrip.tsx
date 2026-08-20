import type { GalleryItem } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { HStack, Icon } from '@chakra-ui/react';
import { galleryImageItemToGalleryImage, isGalleryImageItem, toGalleryItemRef } from '@features/gallery/contracts';
import { Button, IconButton, Tooltip } from '@platform/ui';
import {
  CopyIcon,
  EllipsisVerticalIcon,
  FileJsonIcon,
  ImagesIcon,
  PencilIcon,
  StarIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewDensity } from './previewDensity';

/**
 * The header's always-visible action row: quick verbs one click away, plus an
 * "image actions" dropdown that opens the full right-click context menu
 * (anchored under the button) — one source of truth for every image verb,
 * mirroring the legacy viewer's menu button. Compact densities collapse to
 * star + the dropdown.
 *
 * The strip earns its space by holding what the dropdown cannot reach in one
 * click. Copy and Download sat here *and* in the dropdown's quick row, so they
 * cost a slot each to save nothing; editing on canvas — the one verb that
 * moves the image into actual work — had no button at all. Copy and Download
 * stay one click away in the dropdown.
 */
export const PreviewActionStrip = ({
  actions,
  density,
  isVideoFrameCopyAvailable = false,
  item,
  onCopyCurrentFrame,
  onOpenDetails,
  onOpenMenu,
}: {
  actions: ImageActions;
  density: PreviewDensity;
  isVideoFrameCopyAvailable?: boolean;
  item: GalleryItem;
  onCopyCurrentFrame?: () => void;
  onOpenDetails?: () => void;
  /** Opens the view's image context menu at viewport coordinates. */
  onOpenMenu: ((x: number, y: number) => void) | null;
}) => {
  const { t } = useTranslation();
  const image = isGalleryImageItem(item) ? galleryImageItemToGalleryImage(item) : null;
  const toggleStar = useCallback(
    () => void actions.setItemsStarred([toGalleryItemRef(item)], !item.starred),
    [actions, item]
  );
  const selectForCompare = useCallback(() => image && actions.selectForCompare(image), [actions, image]);
  const editOnCanvas = useCallback(() => image && void actions.sendToCanvas([image], 'raster'), [actions, image]);
  const copyCurrentFrame = useCallback(() => onCopyCurrentFrame?.(), [onCopyCurrentFrame]);
  const openDetails = useCallback(() => onOpenDetails?.(), [onOpenDetails]);
  const openMenu = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();

      onOpenMenu?.(rect.left, rect.bottom + 4);
    },
    [onOpenMenu]
  );
  const itemKindLabel = item.kind === 'video' ? 'video' : 'image';
  const starLabel = `${item.starred ? 'Unstar' : 'Star'} ${itemKindLabel}`;
  const starButton = (
    <Tooltip content={starLabel}>
      <IconButton aria-label={starLabel} color="fg.muted" size="2xs" variant="ghost" onClick={toggleStar}>
        <Icon as={StarIcon} boxSize="3.5" fill={item.starred ? 'currentColor' : 'none'} />
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
      {/* First and worded, not another glyph in the row: it is the strip's one
          verb that moves the image into actual work. "Edit" like legacy's
          button; the aria-label keeps the destination. Videos have no canvas
          destination, so it simply is not offered for them — the same guard
          the context menu's image-only branch uses. */}
      {image ? (
        <Button
          aria-label={t('widgets.preview.editOnCanvas')}
          color="fg.muted"
          size="2xs"
          variant="ghost"
          onClick={editOnCanvas}
        >
          <Icon as={PencilIcon} boxSize="3.5" />
          {t('common.edit')}
        </Button>
      ) : null}
      {starButton}
      {image ? <StripIconButton icon={ImagesIcon} label="Select for Compare" onClick={selectForCompare} /> : null}
      {item.kind === 'video' && onCopyCurrentFrame ? (
        <StripIconButton
          disabled={!isVideoFrameCopyAvailable}
          icon={CopyIcon}
          label={t('widgets.preview.copyCurrentFrame')}
          onClick={copyCurrentFrame}
        />
      ) : null}
      {item.kind === 'video' && onOpenDetails ? (
        <StripIconButton icon={FileJsonIcon} label={t('widgets.preview.videoDetails')} onClick={openDetails} />
      ) : null}
      {menuButton}
    </HStack>
  );
};

const StripIconButton = ({
  disabled,
  icon,
  label,
  onClick,
}: {
  disabled?: boolean;
  icon: LucideIcon;
  label: string;
  onClick: () => void;
}) => (
  <Tooltip content={label}>
    <IconButton aria-label={label} color="fg.muted" disabled={disabled} size="2xs" variant="ghost" onClick={onClick}>
      <Icon as={icon} boxSize="3.5" />
    </IconButton>
  </Tooltip>
);

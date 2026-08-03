import type { GalleryItem, GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryThumbnailFit } from '@features/gallery/core/settings';

import { Badge, Box } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import { formatGalleryVideoDuration, toGalleryItemRef } from '@features/gallery/core/items';
import { IconButton } from '@platform/ui/Button';
import { PlayIcon, StarIcon } from 'lucide-react';
import { useCallback, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';

import { getGalleryItemDragData, getGalleryItemDragId } from './galleryDnd';

const THUMBNAIL_HOVER_CSS = {
  '&:focus-within': { outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' },
  '&:hover .gallery-thumb-overlay, &:focus-within .gallery-thumb-overlay': { opacity: 1 },
} as const;

const PREVIEW_IMAGE_STYLE = {
  borderRadius: '0.375rem',
  boxShadow: '0 8px 24px rgb(0 0 0 / 45%)',
  height: '100%',
  objectFit: 'cover',
  width: '100%',
} as const;

const THUMBNAIL_BUTTON_STYLE = {
  background: 'transparent',
  border: 0,
  cursor: 'pointer',
  display: 'block',
  height: '100%',
  inset: 0,
  minWidth: 0,
  outline: 'none',
  padding: 0,
  position: 'absolute',
  width: '100%',
} as const;

const GalleryThumbnail = ({
  alwaysShowDimensions,
  compareRole,
  dragItems,
  dragScope,
  fit,
  isPrimary,
  isSelected,
  item,
  onClick,
  onContextMenu,
  onToggleStarred,
}: {
  alwaysShowDimensions: boolean;
  compareRole: string | null;
  dragItems: GalleryItemRef[];
  /** Separates this gallery's drags from another instance showing the same item. */
  dragScope: string;
  fit: GalleryThumbnailFit;
  isPrimary: boolean;
  isSelected: boolean;
  item: GalleryItem;
  onClick: (item: GalleryItem, event: MouseEvent) => void;
  onContextMenu: (item: GalleryItem, x: number, y: number) => void;
  onToggleStarred: (item: GalleryItem) => void;
}) => {
  const { t } = useTranslation();
  const isCompared = compareRole !== null;
  const duration = item.kind === 'video' ? formatGalleryVideoDuration(item.durationSeconds) : null;

  const { isDragging, listeners, setNodeRef, transform } = useDraggable({
    data: getGalleryItemDragData(dragItems),
    id: getGalleryItemDragId(toGalleryItemRef(item), 'gallery-grid', dragScope),
  });

  // The preview is portalled from where the tile started rather than moved in
  // place: the grid scrolls inside `overflow: hidden` and its virtual rows carry
  // their own transform, so a moved tile is clipped as soon as it leaves the
  // grid — which is most of the way to any board.
  const tileRef = useRef<HTMLDivElement | null>(null);
  const [dragOrigin, setDragOrigin] = useState<DOMRect | null>(null);

  useLayoutEffect(() => {
    setDragOrigin(isDragging ? (tileRef.current?.getBoundingClientRect() ?? null) : null);
  }, [isDragging]);

  const setTileRef = useCallback(
    (node: HTMLDivElement | null) => {
      tileRef.current = node;
      setNodeRef(node);
    },
    [setNodeRef]
  );

  const previewStyle = useMemo(
    () =>
      dragOrigin
        ? ({
            height: `${String(dragOrigin.height)}px`,
            left: `${String(dragOrigin.left)}px`,
            pointerEvents: 'none',
            position: 'fixed',
            top: `${String(dragOrigin.top)}px`,
            // Translate only: the full transform carries dnd-kit's scale factors,
            // which squash the preview to whatever it is hovering over.
            transform: CSS.Translate.toString(transform),
            width: `${String(dragOrigin.width)}px`,
            zIndex: 1500,
          } as const)
        : null,
    [dragOrigin, transform]
  );

  const imageStyle = useMemo(
    () =>
      ({
        display: 'block',
        height: '100%',
        inset: 0,
        maxWidth: 'none',
        objectFit: fit === 'aspect' ? 'contain' : 'cover',
        position: 'absolute',
        width: '100%',
      }) as const,
    [fit]
  );

  const handleContextMenu = useCallback(
    (event: MouseEvent) => {
      event.preventDefault();
      onContextMenu(item, event.clientX, event.clientY);
    },
    [item, onContextMenu]
  );

  const handleClick = useCallback((event: MouseEvent) => onClick(item, event), [item, onClick]);

  const handleActivationKeyDown = useCallback((event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.stopPropagation();
    }
  }, []);

  const handleToggleStarred = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation();

      onToggleStarred(item);
    },
    [item, onToggleStarred]
  );

  return (
    <Box
      ref={setTileRef}
      {...listeners}
      aspectRatio={1}
      bg="bg"
      borderColor={isSelected || isCompared ? 'accent.solid' : 'border.subtle'}
      borderWidth="2px"
      boxShadow={isCompared ? 'inset 0 0 0 1px {colors.accent.solid}' : undefined}
      css={THUMBNAIL_HOVER_CSS}
      minW="0"
      opacity={isDragging ? 0.4 : undefined}
      overflow="hidden"
      position="relative"
      role="listitem"
      rounded="md"
      touchAction="none"
      w="full"
      onContextMenu={handleContextMenu}
    >
      <button
        aria-current={isPrimary ? 'true' : undefined}
        aria-label={
          item.kind === 'video'
            ? t('widgets.gallery.selectVideoForPreview', { duration, name: item.name })
            : t('widgets.gallery.selectImageForPreview', { name: item.name })
        }
        aria-pressed={isSelected}
        style={THUMBNAIL_BUTTON_STYLE}
        type="button"
        onClick={handleClick}
        onKeyDown={handleActivationKeyDown}
      >
        <img
          alt={item.name}
          decoding={item.kind === 'video' ? 'async' : undefined}
          draggable={false}
          src={item.thumbnailUrl || item.fullUrl}
          style={imageStyle}
        />
      </button>
      {compareRole && (
        <Badge
          insetInlineStart="1"
          pointerEvents="none"
          position="absolute"
          size="xs"
          top="1"
          variant="solid"
          zIndex="1"
        >
          {compareRole}
        </Badge>
      )}
      <IconButton
        aria-label={
          item.starred
            ? t('widgets.gallery.unstarImage', { name: item.name })
            : t('widgets.gallery.starImage', { name: item.name })
        }
        className="gallery-thumb-overlay"
        colorPalette={item.starred ? 'yellow' : 'gray'}
        insetInlineEnd="1"
        opacity={item.starred ? 1 : 0}
        position="absolute"
        size="2xs"
        top="1"
        transition="opacity var(--wb-motion-duration-medium) ease"
        variant="solid"
        zIndex="1"
        onClick={handleToggleStarred}
      >
        <StarIcon fill={item.starred ? 'currentColor' : 'none'} />
      </IconButton>
      {item.kind === 'image' && item.width > 0 && item.height > 0 && (
        <Badge
          bottom="1"
          className="gallery-thumb-overlay"
          insetInlineStart="1"
          opacity={alwaysShowDimensions ? 1 : 0}
          pointerEvents="none"
          position="absolute"
          size="xs"
          transition="opacity var(--wb-motion-duration-medium) ease"
          variant="solid"
          zIndex="1"
        >
          {item.width}x{item.height}
        </Badge>
      )}
      {duration !== null && (
        <Badge
          bottom="1"
          display="flex"
          fontVariantNumeric="tabular-nums"
          gap="1"
          insetInlineStart="1"
          opacity={1}
          pointerEvents="none"
          position="absolute"
          size="xs"
          transition="opacity var(--wb-motion-duration-medium) ease"
          variant="solid"
          zIndex="1"
        >
          <PlayIcon aria-hidden="true" fill="currentColor" />
          {duration}
        </Badge>
      )}
      {previewStyle
        ? createPortal(
            <div aria-hidden="true" style={previewStyle}>
              <img alt="" src={item.thumbnailUrl || item.fullUrl} style={PREVIEW_IMAGE_STYLE} />
            </div>,
            document.body
          )
        : null}
    </Box>
  );
};

/**
 * Resolves the drag payload per item so the grid can hand down one stable
 * callback rather than an array prop that changes identity every render.
 */
export const GalleryThumbnailCell = ({
  getDragItems,
  item,
  ...props
}: Omit<Parameters<typeof GalleryThumbnail>[0], 'dragItems'> & {
  getDragItems: (item: GalleryItem) => GalleryItemRef[];
}) => {
  const dragItems = useMemo(() => getDragItems(item), [getDragItems, item]);

  return <GalleryThumbnail {...props} dragItems={dragItems} item={item} />;
};

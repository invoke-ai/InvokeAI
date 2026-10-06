import type { GalleryItem, GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryThumbnailFit } from '@features/gallery/core/settings';

import { Badge, chakra } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import { formatGalleryVideoDuration, toGalleryItemRef } from '@features/gallery/core/items';
import { IconButton } from '@platform/ui/Button';
import { StarIcon } from 'lucide-react';
import { useCallback, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';

import { getGalleryItemDragData, getGalleryItemDragId } from './galleryDnd';
import { GalleryTileFrame } from './GalleryTileFrame';

/** Desaturate armed and active touch drags, including the portalled preview. */
const THUMBNAIL_DRAG_CSS = { filter: 'saturate(0)' } as const;
const THUMBNAIL_ARMED_CSS = { '&[data-drag-armed=true]': { filter: 'saturate(0)' } } as const;

/** Modified chords are app hotkeys (Mod+Enter invokes), never tile activation or a keyboard drag. */
const isModifiedKey = (event: KeyboardEvent) => event.ctrlKey || event.metaKey || event.altKey || event.shiftKey;

const PREVIEW_IMAGE_STYLE = {
  borderRadius: '0.375rem',
  boxShadow: '0 8px 24px rgb(0 0 0 / 45%)',
  filter: 'saturate(0)',
  height: '100%',
  objectFit: 'cover',
  width: '100%',
} as const;

/** Leaves room for the star button in the opposite corner. */
const LABEL_MAX_WIDTH = 'calc(100% - 2.25rem)';
/**
 * A label that resolves mid-hover mounts already revealed; fade it in like the overlays it joins. Important
 * because the tile's hover reveal outranks a single-class rule.
 */
const LABEL_CSS = { '@starting-style': { opacity: '0 !important' } } as const;

const THUMBNAIL_BUTTON_STYLE = {
  background: 'transparent',
  border: 0,
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
  getItemLabel,
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
  /** Null while image-map labels are unavailable. */
  getItemLabel: ((item: GalleryItemRef) => Promise<string | null>) | null;
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
  const dragListeners = useMemo(
    () =>
      listeners && {
        ...listeners,
        onKeyDown: (event: KeyboardEvent) => {
          if (!isModifiedKey(event)) {
            listeners.onKeyDown?.(event);
          }
        },
      },
    [listeners]
  );

  // Portal the drag preview to escape the grid's overflow clipping and transformed virtual rows.
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

      // A long-press that armed or started a touch drag is drag intent, not
      // menu intent: the native menu is already suppressed by the sensor, and
      // the app menu must not open under a gesture that is about to move the
      // image.
      if (!isDragging) {
        onContextMenu(item, event.clientX, event.clientY);
      }
    },
    [isDragging, item, onContextMenu]
  );

  const handleClick = useCallback((event: MouseEvent) => onClick(item, event), [item, onClick]);

  const handleActivationKeyDown = useCallback((event: KeyboardEvent<HTMLButtonElement>) => {
    if (!isModifiedKey(event) && (event.key === 'Enter' || event.key === ' ')) {
      event.stopPropagation();
    }
  }, []);

  // Fetched on reveal rather than per rendered tile: labels cost a request each, and the cache makes repeat
  // reveals free while still picking up a rebuilt vocabulary.
  const [label, setLabel] = useState<string | null>(null);
  const handleRevealLabel = useCallback(() => {
    if (getItemLabel) {
      void getItemLabel(toGalleryItemRef(item)).then(setLabel);
    }
  }, [getItemLabel, item]);

  const handleToggleStarred = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation();

      onToggleStarred(item);
    },
    [item, onToggleStarred]
  );

  return (
    <GalleryTileFrame
      ref={setTileRef}
      {...dragListeners}
      alwaysShowDimensions={alwaysShowDimensions}
      boxShadow={isCompared ? 'inset 0 0 0 1px {colors.accent.solid}' : undefined}
      css={isDragging ? THUMBNAIL_DRAG_CSS : THUMBNAIL_ARMED_CSS}
      isSelected={isSelected || isCompared}
      item={item}
      opacity={isDragging ? 0.4 : undefined}
      role="listitem"
      // Allow touch panning; the hold sensor yields to scrolling and arms drag only after a sustained hold.
      touchAction="pan-y"
      onContextMenu={handleContextMenu}
      onFocus={handleRevealLabel}
      onPointerEnter={handleRevealLabel}
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
      {/* The compare role owns this corner while comparing. */}
      {label && getItemLabel && !compareRole ? (
        <Badge
          // Hover-dependent and heuristic: keep it out of the tile's accessible text, which the name already covers.
          aria-hidden="true"
          className="gallery-thumb-overlay"
          css={LABEL_CSS}
          insetInlineStart="1"
          maxW={LABEL_MAX_WIDTH}
          opacity={0}
          pointerEvents="none"
          position="absolute"
          top="1"
          transition="opacity var(--wb-motion-duration-medium) ease"
          variant="solid"
          zIndex="1"
        >
          <chakra.span minW="0" truncate>
            {label}
          </chakra.span>
        </Badge>
      ) : null}
      {compareRole && (
        <Badge insetInlineStart="1" pointerEvents="none" position="absolute" top="1" variant="solid" zIndex="1">
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
        size="sm"
        top="1"
        transition="opacity var(--wb-motion-duration-medium) ease"
        variant="solid"
        zIndex="1"
        onClick={handleToggleStarred}
      >
        <StarIcon fill={item.starred ? 'currentColor' : 'none'} />
      </IconButton>
      {previewStyle
        ? createPortal(
            <div aria-hidden="true" style={previewStyle}>
              <img alt="" src={item.thumbnailUrl || item.fullUrl} style={PREVIEW_IMAGE_STYLE} />
            </div>,
            document.body
          )
        : null}
    </GalleryTileFrame>
  );
};

/** Resolve payloads per item so the grid passes one stable callback rather than recreated arrays. */
export const GalleryThumbnailCell = ({
  getDragItems,
  item,
  onClick,
  selectionPage,
  ...props
}: Omit<Parameters<typeof GalleryThumbnail>[0], 'dragItems' | 'onClick'> & {
  getDragItems: (item: GalleryItem) => GalleryItemRef[];
  onClick: (item: GalleryItem, event: MouseEvent, selectionPage?: number) => void;
  selectionPage?: number;
}) => {
  const dragItems = useMemo(() => getDragItems(item), [getDragItems, item]);
  const handleClick = useCallback(
    (clickedItem: GalleryItem, event: MouseEvent) => onClick(clickedItem, event, selectionPage),
    [onClick, selectionPage]
  );

  return <GalleryThumbnail {...props} dragItems={dragItems} item={item} onClick={handleClick} />;
};

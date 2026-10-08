import type { GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryThumbnailFit } from '@features/gallery/core/settings';

import { Badge, chakra } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import { formatGalleryVideoDuration, toGalleryItemKey, toGalleryItemRef } from '@features/gallery/core/items';
import {
  getGalleryThumbnailRevision,
  getRefreshedGalleryThumbnailUrl,
  subscribeGalleryThumbnailRevision,
} from '@features/gallery/queries';
import { IconButton } from '@platform/ui/Button';
import { StarIcon } from 'lucide-react';
import {
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type KeyboardEvent,
  type MouseEvent,
} from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';

import { getGalleryItemDragData, getGalleryItemDragId } from './galleryDnd';
import { GalleryTileFrame } from './GalleryTileFrame';

/** Desaturate armed and active touch drags, including the portalled preview. */
/** The grid's one thumbnail Tab stop: the tile keyboard focus returns to. */
export const GALLERY_TAB_STOP_SELECTOR = 'button[data-gallery-item-key][tabindex="0"]';

const THUMBNAIL_DRAG_CSS = { filter: 'saturate(0)' } as const;
const THUMBNAIL_ARMED_CSS = { '&[data-drag-armed=true]': { filter: 'saturate(0)' } } as const;

/**
 * Modified chords are app hotkeys (Mod+Enter invokes), never tile activation or a keyboard drag; modified clicks are
 * selection gestures, never an open.
 */
const isModified = (event: KeyboardEvent | MouseEvent) =>
  event.ctrlKey || event.metaKey || event.altKey || event.shiftKey;

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
  isTabStop,
  item,
  onClick,
  onContextMenu,
  onFocusLost,
  onOpen,
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
  /** The grid's one thumbnail Tab stop; the other thumbnails are reached with the arrow keys. */
  isTabStop: boolean;
  item: GalleryItem;
  onClick: (item: GalleryItem, event: MouseEvent) => void;
  onContextMenu: (item: GalleryItem, x: number, y: number) => void;
  /** Runs after the tile left the document holding focus, once the grid has committed what replaced it. */
  onFocusLost: (itemKey: GalleryItemKey) => void;
  /** Double-click or Enter: show the item in the centre Preview. */
  onOpen: (item: GalleryItem) => void;
  onToggleStarred: (item: GalleryItem) => void;
}) => {
  const { t } = useTranslation();
  const thumbnailRevision = useSyncExternalStore(
    subscribeGalleryThumbnailRevision,
    getGalleryThumbnailRevision,
    getGalleryThumbnailRevision
  );
  const thumbnailUrl = useMemo(
    () =>
      item.kind === 'image'
        ? getRefreshedGalleryThumbnailUrl(item.thumbnailUrl || item.fullUrl, thumbnailRevision)
        : item.thumbnailUrl || item.fullUrl,
    [item.fullUrl, item.kind, item.thumbnailUrl, thumbnailRevision]
  );
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
          if (!isModified(event)) {
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

  const itemKey = toGalleryItemKey(item);
  // Covers focus anywhere in the tile, its star toggle included: a star click can move the item between sections.
  const setTileRef = useCallback(
    (node: HTMLDivElement | null) => {
      tileRef.current = node;
      setNodeRef(node);

      if (!node) {
        return;
      }

      return () => {
        if (node.contains(document.activeElement)) {
          queueMicrotask(() => onFocusLost(itemKey));
        }
        tileRef.current = null;
        setNodeRef(null);
      };
    },
    [itemKey, onFocusLost, setNodeRef]
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
  // The clicks before it have already selected the item.
  const handleDoubleClick = useCallback(
    (event: MouseEvent) => {
      if (!isModified(event)) {
        onOpen(item);
      }
    },
    [item, onOpen]
  );

  // Enter opens and Space selects; neither starts a keyboard drag. Enter's default would click (select) first.
  const handleActivationKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      if (isModified(event) || (event.key !== 'Enter' && event.key !== ' ')) {
        return;
      }

      event.stopPropagation();

      if (event.key === 'Enter') {
        event.preventDefault();

        if (!event.repeat) {
          onOpen(item);
        }
      }
    },
    [item, onOpen]
  );

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
        // Enter opens the item in Preview; Space keeps the button's own activation, which selects.
        aria-keyshortcuts="Enter"
        aria-label={
          item.kind === 'video'
            ? t('widgets.gallery.selectVideoForPreview', { duration, name: item.name })
            : t('widgets.gallery.selectImageForPreview', { name: item.name })
        }
        aria-pressed={isSelected}
        data-gallery-item-key={itemKey}
        style={THUMBNAIL_BUTTON_STYLE}
        tabIndex={isTabStop ? 0 : -1}
        type="button"
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onKeyDown={handleActivationKeyDown}
      >
        <img
          alt={item.name}
          decoding={item.kind === 'video' ? 'async' : undefined}
          draggable={false}
          src={thumbnailUrl}
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
        // Pointer-only: the star hotkey toggles the selection, which keyboard focus follows.
        tabIndex={-1}
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
              <img alt="" src={thumbnailUrl} style={PREVIEW_IMAGE_STYLE} />
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
  ...props
}: Omit<Parameters<typeof GalleryThumbnail>[0], 'dragItems'> & {
  getDragItems: (item: GalleryItem) => GalleryItemRef[];
}) => {
  const dragItems = useMemo(() => getDragItems(item), [getDragItems, item]);

  return <GalleryThumbnail {...props} dragItems={dragItems} item={item} />;
};

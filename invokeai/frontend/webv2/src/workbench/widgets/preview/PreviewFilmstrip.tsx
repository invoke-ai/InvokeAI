import type { GalleryItem, GalleryItemKey } from '@features/gallery';
import type { QueueProgressSession } from '@features/queue/contracts';

import { Box, HStack, Icon, ProgressCircle, Skeleton } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { toGalleryItemKey, toGalleryItemRef } from '@features/gallery/contracts';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { getDeterminateProgressPercent } from '@features/queue/contracts';
import { useDeviceLabel } from '@features/queue/devices';
import { useItemProgress, useQueueItemProgressImage } from '@features/queue/react';
import { Scrollable } from '@platform/ui/Scrollable';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import { progressImageToStreamingSource } from '@platform/ui/streaming-image/streamingImageSource';
import { CheckIcon, HourglassIcon, PinIcon } from 'lucide-react';
import { useCallback, useMemo, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewDensity } from './previewDensity';

/**
 * Render board thumbnails below the stage with live GPU slots leading in gallery order and settling into result
 * thumbnails in place. Reserve height outside media; item drag payloads work with existing gallery targets.
 */

export const PreviewFilmstrip = ({
  density,
  followedSessionId = null,
  isSessionPinned = false,
  items,
  selectedItemKey,
  sessions = EMPTY_SESSIONS,
  onCompare,
  onContextMenu,
  onFollowSession,
  onSelect,
  onUnpinSession,
  shouldAntialiasLiveImage = true,
}: {
  density: PreviewDensity;
  /** The live session on the stage, if the preview is following one. */
  followedSessionId?: string | null;
  /** Whether that session was pinned by hand; clicking it again unpins. */
  isSessionPinned?: boolean;
  items: GalleryItem[];
  selectedItemKey: GalleryItemKey | null;
  /** In-progress slots in the gallery's order, leading the row. */
  sessions?: readonly QueueProgressSession[];
  onFollowSession?: (sessionId: string) => void;
  onUnpinSession?: () => void;
  /** The project's live-frame antialiasing setting, mirrored on the live thumbs. */
  shouldAntialiasLiveImage?: boolean;
  /** Alt-click: arm this thumb for comparison against the selection. */
  onCompare?: (item: GalleryItem) => void;
  /** Right-click: the image context menu for this thumb, at viewport coordinates. */
  onContextMenu?: (item: GalleryItem, x: number, y: number) => void;
  onSelect: (item: GalleryItem) => void;
}) => {
  const thumbSize = density === 'full' ? '12' : '8';

  // Show the strip for any live session, even on an otherwise empty board.
  if (items.length < 2 && sessions.length === 0) {
    return null;
  }

  return (
    // Set explicit ScrollArea height and inline-size containment so the strip neither consumes widget height nor
    // expands its width.
    <Scrollable
      bg="bg.subtle"
      borderColor="border.subtle"
      borderTopWidth="1px"
      contentProps={FILMSTRIP_CONTENT_PROPS}
      css={FILMSTRIP_CONTAIN_CSS}
      data-preview-filmstrip
      flexShrink={0}
      h={density === 'full' ? '3.75rem' : '2.75rem'}
      minW="0"
      orientation="horizontal"
      px="2"
      w="full"
    >
      <HStack align="center" gap="1" h="full">
        {sessions.map((session) => (
          <FilmstripLiveThumb
            key={session.id}
            isFollowed={session.id === followedSessionId}
            isPinned={isSessionPinned && session.id === followedSessionId}
            session={session}
            shouldAntialiasLiveImage={shouldAntialiasLiveImage}
            size={thumbSize}
            onFollow={onFollowSession}
            onUnpin={onUnpinSession}
          />
        ))}
        {items.map((item) => {
          const itemKey = toGalleryItemKey(item);

          return (
            <FilmstripThumb
              key={itemKey}
              item={item}
              isSelected={itemKey === selectedItemKey}
              size={thumbSize}
              onCompare={onCompare}
              onContextMenu={onContextMenu}
              onSelect={onSelect}
            />
          );
        })}
      </HStack>
    </Scrollable>
  );
};

/**
 * Show live/settling/queued slots with progress, device, and selected/pinned state; only running slots can be
 * followed.
 */
const FilmstripLiveThumb = ({
  isFollowed,
  isPinned,
  onFollow,
  onUnpin,
  session,
  shouldAntialiasLiveImage,
  size,
}: {
  isFollowed: boolean;
  isPinned: boolean;
  onFollow?: (sessionId: string) => void;
  onUnpin?: () => void;
  session: QueueProgressSession;
  shouldAntialiasLiveImage: boolean;
  size: string;
}) => {
  const { t } = useTranslation();
  const image = useQueueItemProgressImage(session.queueItemId, session.itemIndex);
  const progress = useItemProgress(session.backendItemId);
  const deviceLabel = useDeviceLabel(progress?.device);
  const percentage = getDeterminateProgressPercent(progress?.percentage);
  const isRunning = session.state === 'running';
  const name =
    session.itemCount > 1
      ? t('widgets.gallery.progressSession', {
          index: session.itemIndex,
          name: session.label,
          total: session.itemCount,
        })
      : session.label;
  const status =
    session.state === 'queued'
      ? t('widgets.gallery.progressQueued')
      : session.state === 'settling'
        ? t('widgets.gallery.progressSettling')
        : percentage !== null
          ? `${percentage}%`
          : progress?.message || t('widgets.gallery.progressPreparing');
  const label = [
    name,
    ...(deviceLabel ? [t('widgets.queue.device.shortLabel', { index: deviceLabel.index })] : []),
    status,
  ].join(' · ');
  const handleClick = useCallback(() => {
    if (!isRunning) {
      return;
    }
    if (isPinned) {
      onUnpin?.();
    } else {
      onFollow?.(session.id);
    }
  }, [isPinned, isRunning, onFollow, onUnpin, session.id]);
  // Keep the followed slot visible and restore preview focus when its thumb disappears.
  const thumbRef = useCallback(
    (node: HTMLElement | null) => {
      if (!node) {
        return;
      }
      if (isFollowed) {
        node.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
      const region = node.closest<HTMLElement>('[role="region"]');

      // React re-runs this cleanup when `isFollowed` flips while the node is
      // still mounted — exactly the click that focused it — so only a node
      // that really left the document gives focus away.
      return () => {
        if (document.activeElement === node) {
          queueMicrotask(() => {
            if (!node.isConnected) {
              region?.focus({ preventScroll: true });
            }
          });
        }
      };
    },
    [isFollowed]
  );

  return (
    <Box
      ref={thumbRef}
      as="button"
      aria-current={isFollowed || undefined}
      aria-disabled={!isRunning}
      aria-label={label}
      aria-pressed={isRunning ? isPinned : undefined}
      borderColor={isFollowed ? 'accent.solid' : 'border.subtle'}
      borderWidth="2px"
      boxSize={size}
      data-preview-live-pinned={isPinned || undefined}
      data-preview-live-thumb={session.id}
      flexShrink={0}
      overflow="hidden"
      position="relative"
      rounded="sm"
      tabIndex={isRunning ? 0 : -1}
      onClick={handleClick}
    >
      <StreamingImageFrame
        fit="cover"
        h="full"
        liveImage={progressImageToStreamingSource(image)}
        shouldAntialiasLiveImage={shouldAntialiasLiveImage}
        w="full"
      >
        {session.state === 'queued' ? <Box bg="bg.subtle" h="full" w="full" /> : <Skeleton h="full" w="full" />}
      </StreamingImageFrame>
      <Box
        bg="bg/85"
        bottom="0.5"
        display="flex"
        p="0.5"
        pointerEvents="none"
        position="absolute"
        right="0.5"
        rounded="full"
      >
        {isPinned ? (
          <Icon aria-hidden as={PinIcon} boxSize="3" />
        ) : isRunning ? (
          <ProgressCircle.Root aria-hidden size="xs" value={percentage}>
            <ProgressCircle.Circle>
              <ProgressCircle.Track />
              <ProgressCircle.Range />
            </ProgressCircle.Circle>
          </ProgressCircle.Root>
        ) : (
          <Icon aria-hidden as={session.state === 'queued' ? HourglassIcon : CheckIcon} boxSize="3" />
        )}
      </Box>
    </Box>
  );
};

const FilmstripThumb = ({
  item,
  isSelected,
  size,
  onCompare,
  onContextMenu,
  onSelect,
}: {
  item: GalleryItem;
  isSelected: boolean;
  size: string;
  onCompare?: (item: GalleryItem) => void;
  onContextMenu?: (item: GalleryItem, x: number, y: number) => void;
  onSelect: (item: GalleryItem) => void;
}) => {
  const itemRef = useMemo(() => toGalleryItemRef(item), [item]);
  const dragData = useMemo(() => getGalleryItemDragData([itemRef]), [itemRef]);
  const thumbnailSrc = item.thumbnailUrl || (item.kind === 'image' ? item.fullUrl : null);
  const { isDragging, listeners, setNodeRef } = useDraggable({
    data: dragData,
    id: getGalleryItemDragId(itemRef, 'preview-filmstrip'),
  });
  const handleClick = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      if (event.altKey && onCompare) {
        onCompare(item);
        return;
      }

      onSelect(item);
    },
    [item, onCompare, onSelect]
  );
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      if (onContextMenu) {
        event.preventDefault();
        onContextMenu(item, event.clientX, event.clientY);
      }
    },
    [item, onContextMenu]
  );
  const scrollIntoView = useCallback(
    (node: HTMLElement | null) => {
      setNodeRef(node);

      if (node && isSelected) {
        node.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
    },
    [isSelected, setNodeRef]
  );

  return (
    <Box
      ref={scrollIntoView}
      {...listeners}
      as="button"
      aria-current={isSelected || undefined}
      aria-label={item.kind === 'video' ? `Video ${item.name}` : item.name}
      borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
      borderWidth="2px"
      boxSize={size}
      css={isDragging ? FILMSTRIP_THUMB_DRAG_CSS : FILMSTRIP_THUMB_ARMED_CSS}
      flexShrink={0}
      opacity={isDragging ? 0.4 : undefined}
      overflow="hidden"
      position="relative"
      rounded="sm"
      // Allow horizontal touch pan; sustained holds still activate dragging.
      touchAction="pan-x"
      onClick={handleClick}
      onContextMenu={onContextMenu ? handleContextMenu : undefined}
    >
      {thumbnailSrc ? (
        <img alt={item.name} loading="lazy" src={thumbnailSrc} style={FILMSTRIP_IMG_STYLE} />
      ) : (
        <Box aria-hidden bg="bg.muted" h="full" w="full" />
      )}
    </Box>
  );
};

const EMPTY_SESSIONS: readonly QueueProgressSession[] = [];

const FILMSTRIP_CONTAIN_CSS = { contain: 'inline-size' } as const;

/** Same touch drag cue as the gallery grid: the source thumb desaturates while dragged. */
const FILMSTRIP_THUMB_DRAG_CSS = { filter: 'saturate(0)' } as const;

/** Match Gallery's desaturated data-drag-armed cue before held-touch movement starts dragging. */
const FILMSTRIP_THUMB_ARMED_CSS = { '&[data-drag-armed=true]': { filter: 'saturate(0)' } } as const;

// The thumb row centers itself with `h="full"`, which needs the content
// wrapper to actually span the viewport height rather than shrink to it.
const FILMSTRIP_CONTENT_PROPS = { h: 'full' } as const;

const FILMSTRIP_IMG_STYLE = {
  display: 'block',
  height: '100%',
  objectFit: 'cover',
  width: '100%',
} as const;

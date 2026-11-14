import type { GalleryThumbnailFit } from '@features/gallery/core/settings';
import type { QueueProgressSession, QueueSourceId } from '@features/queue/contracts';

import { Box, chakra, Icon, Skeleton, Text } from '@chakra-ui/react';
import { getDeterminateProgressPercent } from '@features/queue/contracts';
import { useItemProgress, useQueueItemProgressImage } from '@features/queue/react';
import { ProgressBadge } from '@platform/ui/ProgressBadge';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import { progressImageToStreamingSource } from '@platform/ui/streaming-image/streamingImageSource';
import { Tooltip } from '@platform/ui/Tooltip';
import { ChevronRightIcon, HourglassIcon } from 'lucide-react';
import { useCallback, useEffectEvent, useId, useLayoutEffect, useRef } from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { GalleryProgressLayout } from './galleryGridLayout';

import { GALLERY_TAB_STOP_SELECTOR } from './GalleryThumbnail';
import { GalleryTileMedia, getGalleryTileMediaRatio } from './GalleryTileFrame';
import { useGalleryUi } from './GalleryUiContext';
import { useGalleryWidget } from './GalleryWidgetContext';

export const GalleryProgressSection = ({
  layout,
  getScrollElement,
  offsetTopPx,
}: {
  layout: GalleryProgressLayout;
  getScrollElement(): HTMLDivElement | null;
  /** Height of the pinned content above this section in the shared scroll viewport. */
  offsetTopPx: number;
}) => {
  const { t } = useTranslation();
  const { progressSessions, pinnedProgressSessionId, liveFollowEnabled, followProgressSession } = useGalleryUi();
  const { actions, gallery } = useGalleryWidget();
  const { progressSectionCollapsed, showPendingItems } = gallery.settings;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const contentId = useId();
  const restoreFocus = useCallback(() => {
    const root = rootRef.current;
    const viewport = getScrollElement();
    // The section's own disclosure while it stays, else the selection, else the grid's thumbnail Tab stop — never
    // simply the first thumbnail, which is the first starred one.
    const target =
      root?.querySelector<HTMLButtonElement>('[data-progress-disclosure]') ??
      viewport?.querySelector<HTMLElement>('button[aria-current="true"]') ??
      viewport?.querySelector<HTMLElement>(GALLERY_TAB_STOP_SELECTOR) ??
      viewport?.querySelector<HTMLElement>('[role="button"]') ??
      viewport;
    target?.focus({ preventScroll: true });
  }, [getScrollElement]);
  const disclosureRef = useCallback(
    (element: HTMLButtonElement | null) => {
      if (!element) {
        return;
      }
      return () => {
        if (document.activeElement === element) {
          queueMicrotask(restoreFocus);
        }
      };
    },
    [restoreFocus]
  );
  const toggleCollapsed = useCallback(
    () => actions.updateSettings({ progressSectionCollapsed: !progressSectionCollapsed }),
    [actions, progressSectionCollapsed]
  );
  const visible = showPendingItems && progressSessions.length > 0;

  if (!visible) {
    return null;
  }

  return (
    <Box ref={rootRef} aria-label={t('widgets.gallery.inProgress')} flexShrink={0} minW="0" role="region">
      <chakra.button
        ref={disclosureRef}
        display="flex"
        alignItems="center"
        aria-controls={contentId}
        aria-expanded={!progressSectionCollapsed}
        type="button"
        focusVisibleRing="inside"
        data-progress-disclosure
        gap="1"
        h={`${layout.headerHeight}px`}
        py="0"
        px="1"
        textAlign="start"
        title={t('widgets.gallery.progressShared')}
        w="full"
        onClick={toggleCollapsed}
      >
        <Icon as={ChevronRightIcon} boxSize="3" transform={progressSectionCollapsed ? undefined : 'rotate(90deg)'} />
        <Icon as={HourglassIcon} boxSize="3" />
        <Text fontSize="xs" fontWeight="600" letterSpacing="wide" lineHeight="1" textTransform="uppercase">
          {t('widgets.gallery.inProgress')}
        </Text>
        <Text color="fg.muted" fontSize="md" fontVariantNumeric="tabular-nums">
          · {progressSessions.length}
        </Text>
      </chakra.button>
      <Box id={contentId} hidden={progressSectionCollapsed}>
        {!progressSectionCollapsed ? (
          <GalleryProgressGrid
            sessions={progressSessions}
            layout={layout}
            fit={gallery.settings.thumbnailFit}
            getScrollElement={getScrollElement}
            offsetTopPx={offsetTopPx}
            pinnedSessionId={pinnedProgressSessionId}
            liveFollowEnabled={liveFollowEnabled}
            onFollow={followProgressSession}
            restoreFocus={restoreFocus}
          />
        ) : null}
      </Box>
    </Box>
  );
};

const GalleryProgressGrid = ({
  sessions,
  layout,
  fit,
  getScrollElement,
  offsetTopPx,
  pinnedSessionId,
  liveFollowEnabled,
  onFollow,
  restoreFocus,
}: {
  sessions: QueueProgressSession[];
  layout: GalleryProgressLayout;
  fit: GalleryThumbnailFit;
  getScrollElement(): HTMLDivElement | null;
  offsetTopPx: number;
  pinnedSessionId: string | null;
  liveFollowEnabled: boolean;
  onFollow(id: string, options: { revealPreview: boolean }): void;
  restoreFocus(): void;
}) => {
  const { columns, tileSize, headerHeight, paddingBottom, rowCount, rowHeight } = layout;
  const estimateSize = useCallback(() => rowHeight, [rowHeight]);
  const scrollMargin = offsetTopPx + headerHeight;
  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement,
    estimateSize,
    scrollMargin,
    overscan: 2,
  });
  const measure = useEffectEvent(() => virtualizer.measure());
  useLayoutEffect(() => {
    measure();
  }, [rowHeight, columns]);
  return (
    <Box minW="0" pb={`${paddingBottom}px`}>
      <Box position="relative" h={`${virtualizer.totalSize}px`} w="full">
        {virtualizer.virtualItems.flatMap((row) =>
          sessions.slice(row.index * columns, (row.index + 1) * columns).map((session, column) => (
            <Box
              key={session.id}
              position="absolute"
              top={`${row.start - scrollMargin}px`}
              left={`${column * rowHeight}px`}
            >
              <GalleryProgressTile
                session={session}
                size={tileSize}
                fit={fit}
                selected={liveFollowEnabled && (pinnedSessionId === null || pinnedSessionId === session.id)}
                onFollow={onFollow}
                restoreFocus={restoreFocus}
              />
            </Box>
          ))
        )}
      </Box>
    </Box>
  );
};

/**
 * Like a completed thumbnail: hover tints the square cell, while the border, selection and focus ring wrap the media
 * box sized to the job's target ratio, so the finished item takes the same rect. The progress badge stays on the cell.
 */
const PROGRESS_CELL_CSS = {
  _focusVisible: { outline: 'none' },
  // Queued and settling tiles are inert; only a followable one answers hover.
  '&:not([aria-disabled=true]):hover': { bg: 'bg.hover' },
  '&:focus-visible [data-gallery-tile-media]': {
    outline: '2px solid {colors.accent.solid}',
    outlineOffset: '-2px',
  },
} as const;

/**
 * Sources whose snapshot records the size they will output. Workflows take Generate's size, which their graph need
 * not use, so their tiles stay square and letterbox the live image.
 */
const SIZED_PROGRESS_SOURCES: ReadonlySet<QueueSourceId> = new Set(['canvas', 'generate', 'upscale', 'video']);

const GalleryProgressTile = ({
  session,
  size,
  fit,
  selected,
  onFollow,
  restoreFocus,
}: {
  session: QueueProgressSession;
  size: number;
  fit: GalleryThumbnailFit;
  selected: boolean;
  onFollow(id: string, options: { revealPreview: boolean }): void;
  restoreFocus(): void;
}) => {
  const { t } = useTranslation();
  const { antialiasProgressImages } = useGalleryUi();
  const image = useQueueItemProgressImage(session.queueItemId, session.itemIndex);
  const progress = useItemProgress(session.backendItemId);
  const percentage = getDeterminateProgressPercent(progress?.percentage);
  const label =
    session.itemCount > 1
      ? t('widgets.gallery.progressSession', {
          name: session.label,
          index: session.itemIndex,
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
  const follow = useCallback(() => {
    if (session.state === 'running') {
      onFollow(session.id, { revealPreview: true });
    }
  }, [onFollow, session.id, session.state]);
  const buttonRef = useCallback(
    (element: HTMLButtonElement | null) => {
      if (!element) {
        return;
      }
      return () => {
        if (document.activeElement === element) {
          queueMicrotask(restoreFocus);
        }
      };
    },
    [restoreFocus]
  );

  const isShownSelected = selected && session.state !== 'queued';
  const mediaRatio = SIZED_PROGRESS_SOURCES.has(session.sourceId) ? getGalleryTileMediaRatio(session, fit) : null;

  return (
    <Tooltip content={`${label} · ${status}`}>
      <chakra.button
        ref={buttonRef}
        type="button"
        aria-label={`${label} · ${status}`}
        aria-pressed={isShownSelected}
        aria-disabled={session.state !== 'running'}
        data-gallery-session-id={session.id}
        tabIndex={session.state === 'running' ? 0 : -1}
        alignItems="center"
        aspectRatio={1}
        css={PROGRESS_CELL_CSS}
        display="flex"
        flexShrink={0}
        justifyContent="center"
        minW="0"
        rounded="md"
        textAlign="start"
        position="relative"
        transition="background-color var(--wb-motion-duration-fast) ease"
        w={`${size}px`}
        onClick={follow}
      >
        <GalleryTileMedia
          fit={mediaRatio === null ? 'square' : fit}
          height={session.height}
          isSelected={isShownSelected}
          width={session.width}
        >
          <StreamingImageFrame
            // A box sized to the target ratio is filled exactly; an unsized aspect fit letterboxes in the square.
            fit={fit === 'aspect' && mediaRatio === null ? 'contain' : 'cover'}
            h="full"
            liveImage={progressImageToStreamingSource(image)}
            shouldAntialiasLiveImage={antialiasProgressImages}
            w="full"
          >
            {session.state === 'queued' ? <Box bg="bg.subtle" h="full" w="full" /> : <Skeleton h="full" w="full" />}
          </StreamingImageFrame>
        </GalleryTileMedia>
        <ProgressBadge
          label={status}
          percentage={percentage}
          state={session.state === 'settling' ? 'completed' : session.state}
        />
      </chakra.button>
    </Tooltip>
  );
};

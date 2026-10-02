import type { GalleryThumbnailFit } from '@features/gallery/core/settings';
import type { QueueProgressSession } from '@features/queue/contracts';

import { Box, chakra, Icon, ProgressCircle, Skeleton, Text } from '@chakra-ui/react';
import { getDeterminateProgressPercent } from '@features/queue/contracts';
import { useItemProgress, useQueueItemProgressImage } from '@features/queue/react';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import { progressImageToStreamingSource } from '@platform/ui/streaming-image/streamingImageSource';
import { CheckIcon, ChevronRightIcon, HourglassIcon } from 'lucide-react';
import { useCallback, useEffectEvent, useId, useLayoutEffect, useRef } from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { GalleryProgressLayout } from './galleryGridLayout';

import { SELECTED_TILE_CSS } from './GalleryTileFrame';
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
    const target =
      root?.querySelector<HTMLButtonElement>('[data-progress-disclosure]') ??
      viewport?.querySelector<HTMLElement>('button[aria-current="true"]') ??
      viewport?.querySelector<HTMLElement>('[role="listitem"] button') ??
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
        <Text fontSize="2xs" fontWeight="600" letterSpacing="wide" lineHeight="1" textTransform="uppercase">
          {t('widgets.gallery.inProgress')}
        </Text>
        <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums">
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

  return (
    <chakra.button
      ref={buttonRef}
      type="button"
      focusVisibleRing="inside"
      aria-label={`${label} · ${status}`}
      aria-pressed={isShownSelected}
      aria-disabled={session.state !== 'running'}
      tabIndex={session.state === 'running' ? 0 : -1}
      borderColor={isShownSelected ? 'accent.solid' : 'border.subtle'}
      borderWidth="2px"
      css={isShownSelected ? SELECTED_TILE_CSS : undefined}
      flexShrink={0}
      minW="0"
      overflow="hidden"
      rounded="md"
      textAlign="start"
      title={`${label} · ${status}`}
      position="relative"
      w={`${size}px`}
      onClick={follow}
    >
      <StreamingImageFrame
        aspectRatio={1}
        fit={fit === 'aspect' ? 'contain' : 'cover'}
        liveImage={progressImageToStreamingSource(image)}
        shouldAntialiasLiveImage={antialiasProgressImages}
        w="full"
      >
        {session.state === 'queued' ? <Box bg="bg.subtle" h="full" w="full" /> : <Skeleton h="full" w="full" />}
      </StreamingImageFrame>
      <Box
        position="absolute"
        bottom="1"
        right="1"
        pointerEvents="none"
        bg="bg/85"
        rounded="full"
        p="0.5"
        display="flex"
      >
        {session.state === 'running' ? (
          <ProgressCircle.Root aria-label={status} size="xs" value={percentage}>
            <ProgressCircle.Circle>
              <ProgressCircle.Track />
              <ProgressCircle.Range />
            </ProgressCircle.Circle>
          </ProgressCircle.Root>
        ) : (
          <Icon as={session.state === 'queued' ? HourglassIcon : CheckIcon} boxSize="4" aria-label={status} />
        )}
      </Box>
    </chakra.button>
  );
};

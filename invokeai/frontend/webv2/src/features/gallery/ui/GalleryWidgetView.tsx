import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryItemsFilter } from '@features/gallery/data/queries';
import type { TFunction } from 'i18next';

import { toGalleryItemKey, toGalleryItemRef } from '@features/gallery/core/items';
import { getBoundedRecentImages } from '@features/gallery/core/recentImages';
import { getGallerySettings } from '@features/gallery/core/settings';
import { GALLERY_PAGE_SIZE, galleryItemNamesOptions } from '@features/gallery/data/queries';
import { StatusWidgetChip } from '@platform/ui';
import { useQueryClient } from '@tanstack/react-query';
import { ImageIcon, TriangleAlertIcon } from 'lucide-react';
import { useCallback, useEffect, useEffectEvent, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryReadState, GalleryReadStatus, GalleryStateView } from './galleryStateView';

import { GalleryBoardDragMonitor } from './GalleryBoardDragMonitor';
import { getGallerySparseSelectionPages, mergeGalleryLoadedItems } from './galleryGridLayout';
import { GalleryLayout } from './GalleryLayout';
import { GalleryAnnouncer } from './GalleryLoadError';
import {
  getGalleryAnchoredWindowPage,
  getGalleryPage,
  getGalleryProjectBoardId,
  getGalleryRawSelectedBoardId,
  getGallerySearchTerm,
  getGallerySemanticImageQuery,
  getGalleryStarredOnly,
  getGalleryStateView,
  getGalleryTotalImages,
  getGalleryView,
} from './galleryStateView';
import {
  useGalleryItemActions,
  useGalleryUi,
  type GalleryItemActionContext,
  type GalleryWidgetProps,
  type GalleryWidgetRuntime,
} from './GalleryUiContext';
import {
  GalleryWidgetContext,
  type GalleryActions,
  type GalleryStarredStrip,
  type GalleryWidgetContextValue,
} from './GalleryWidgetContext';
import { useGalleryActions } from './useGalleryActions';
import { useGalleryData, type GalleryData, type GalleryListingState } from './useGalleryData';
import { useGalleryStarredStrip } from './useGalleryStarredStrip';

export const shouldPublishGalleryTotal = ({
  knownTotalImages,
  lastPublishedTotal,
  total,
}: {
  knownTotalImages: number | null;
  lastPublishedTotal: number | null;
  total: number | null;
}): boolean =>
  typeof total === 'number' && Number.isFinite(total) && total !== knownTotalImages && total !== lastPublishedTotal;

/** `count` is null while the total is not known; a failed listing says so instead of counting nothing. */
export const GalleryStatusChip = ({
  count,
  isUnavailable = false,
}: {
  count: number | null;
  isUnavailable?: boolean;
}) => {
  const { t } = useTranslation();

  if (isUnavailable) {
    return (
      <StatusWidgetChip icon={TriangleAlertIcon} tone="error">
        {t('widgets.gallery.statusChipUnavailable')}
      </StatusWidgetChip>
    );
  }

  return (
    <StatusWidgetChip icon={ImageIcon}>
      {count === null ? t('widgets.gallery.statusChipUnknown') : t('widgets.gallery.statusChip', { count })}
    </StatusWidgetChip>
  );
};

/** Keeps query identity local to the gallery consumer that owns it. */
export const useGallerySemanticImageQuery = (value: unknown) =>
  useMemo(() => getGallerySemanticImageQuery({ semanticImageQuery: value }), [value]);

export const GalleryWidgetView = ({ presentation, region, runtime }: GalleryWidgetProps) => {
  const {
    gallery: galleryCommands,
    galleryValues,
    generateValues,
    projectId,
    projectName,
    ItemActionsProvider,
  } = useGalleryUi();
  const galleryView = getGalleryView(galleryValues);
  const searchTerm = getGallerySearchTerm(galleryValues);
  const starredOnly = getGalleryStarredOnly(galleryValues);
  const recentImages = useMemo(() => getBoundedRecentImages(galleryValues.recentImages), [galleryValues.recentImages]);
  const page = getGalleryPage(galleryValues);
  const knownTotalImages = getGalleryTotalImages(galleryValues);
  const settings = getGallerySettings(galleryValues);
  const queryClient = useQueryClient();
  const semanticQuery = useGallerySemanticImageQuery(galleryValues.semanticImageQuery);
  const data = useGalleryData({
    galleryView,
    page,
    projectBoardId: getGalleryProjectBoardId(galleryValues),
    recentImages,
    searchTerm,
    selectedBoardId: getGalleryRawSelectedBoardId(galleryValues),
    semanticQuery,
    settings,
    // The grid partitions: starred items live in the strip above it.
    starred: starredOnly,
    // Compact bottom chips have no virtualized grid; keep their single page query on the dense listing path.
    sparseViewport: region !== 'bottom' || presentation === 'expanded',
  });

  const { loadMore, selectedBoardId, total } = data;
  // No strip under a ranked result (no starred filter applies), under the
  // starred-only listing (it would repeat the grid), or in a window anchored
  // mid-board (the banner promises a slice, not the top of the board).
  const starredStrip = useGalleryStarredStrip({
    enabled: semanticQuery === null && !starredOnly && getGalleryAnchoredWindowPage(galleryValues) === 0,
    filter: data.filter,
  });
  const gallery = useMemo(
    () => getGalleryStateView(galleryValues, data.boards, data.items, starredStrip.items),
    [data.boards, data.items, galleryValues, starredStrip.items]
  );
  const loadedItems = useMemo(
    () => mergeGalleryLoadedItems(starredStrip.items, gallery.items),
    [gallery.items, starredStrip.items]
  );
  const sparseSelectionPages = useMemo(
    () =>
      data.sparseListing
        ? getGallerySparseSelectionPages({
            itemSlots: data.sparseListing.itemSlots,
            pageOffset: settings.paginationMode === 'paginated' ? page * GALLERY_PAGE_SIZE : 0,
          })
        : null,
    [data.sparseListing, page, settings.paginationMode]
  );
  const getItemSelectionPage = useCallback(
    (item: GalleryItem) => sparseSelectionPages?.get(toGalleryItemKey(item)) ?? page,
    [page, sparseSelectionPages]
  );
  const lastPublishedTotalRef = useRef<number | null>(null);
  const itemActionFilterIdentity = useMemo(() => JSON.stringify(data.filter), [data.filter]);
  const loadOrderedItemRefs = useCallback(
    async (signal: AbortSignal) => {
      signal.throwIfAborted();
      const result = await queryClient.fetchQuery(galleryItemNamesOptions(data.filter));

      signal.throwIfAborted();
      // Navigation order: the strip's starred items, then the listing.
      return [...starredStrip.items.map(toGalleryItemRef), ...result.items];
    },
    [data.filter, queryClient, starredStrip.items]
  );
  const itemActionContextRef = useRef<GalleryItemActionContext | null>(null);
  const galleryLocationRef = useRef({ galleryView, selectedBoardId });

  // In-flight deletion must read the latest rendered filter and selection without an effect-sized stale window.
  // eslint-disable-next-line react/refs
  itemActionContextRef.current = {
    filterIdentity: itemActionFilterIdentity,
    getItemSelectionPage,
    items: gallery.items,
    loadOrderedRefs: loadOrderedItemRefs,
    selectedItemKey: gallery.selectedItemKey,
  };
  // Capture upload destination at launch; judge completion visibility against the latest rendered board and view.
  // eslint-disable-next-line react/refs
  galleryLocationRef.current = { galleryView, selectedBoardId };

  const getItemActionContext = useCallback(() => itemActionContextRef.current, []);
  const getCurrentGalleryLocation = useCallback(() => galleryLocationRef.current, []);

  const actions = useGalleryActions({
    boards: data.boards,
    getCurrentGalleryLocation,
    loadMore,
    selectedBoardId,
  });

  // Publish fetched totals for footer pagination. React only to this instance's total; reacting to shared totals
  // can make simultaneous views dispatch indefinitely.
  const publishGalleryTotal = useEffectEvent((nextTotal: number) => {
    const lastPublishedTotal = lastPublishedTotalRef.current;

    lastPublishedTotalRef.current = nextTotal;

    if (shouldPublishGalleryTotal({ knownTotalImages, lastPublishedTotal, total: nextTotal })) {
      galleryCommands.setPageInfo(nextTotal);
    }
  });

  useEffect(() => {
    if (typeof total !== 'number' || !Number.isFinite(total)) {
      return;
    }

    publishGalleryTotal(total);
  }, [total]);

  // Clamp both paginated pages and infinite anchors after shrinkage so stale positions cannot appear as empty
  // boards.
  useEffect(() => {
    if (total === null) {
      return;
    }

    const maxPage = Math.max(0, Math.ceil(total / GALLERY_PAGE_SIZE) - 1);

    if (page > maxPage) {
      galleryCommands.setPage(maxPage);
    }
  }, [galleryCommands, page, total]);

  if (region === 'bottom' && presentation !== 'expanded') {
    // The listing total is unstarred-only; the strip's total is the rest. Either one unknown leaves no count to state.
    const stripStatus = starredStrip.state.status;
    const count =
      total === null || stripStatus === 'loading' || stripStatus === 'error' ? null : total + starredStrip.total;

    return <GalleryStatusChip count={count} isUnavailable={data.listing.status === 'error'} />;
  }

  return (
    <ItemActionsProvider
      boards={data.boards}
      generateValues={generateValues}
      getItemActionContext={getItemActionContext}
      projectId={projectId}
    >
      <GalleryWidgetContent
        actions={actions}
        boardsState={data.boardsState}
        filter={data.filter}
        gallery={gallery}
        isWindowTruncated={data.isWindowTruncated}
        listing={data.listing}
        loadedItems={loadedItems}
        projectName={projectName}
        region={region}
        runtime={runtime}
        pinRevealIndex={data.pinRevealIndex}
        setVisibleRange={data.setVisibleRange}
        sparseListing={data.sparseListing}
        starredStrip={starredStrip}
      />
    </ItemActionsProvider>
  );
};

const LISTING_NOTICES: Partial<Record<GalleryReadStatus, string>> = {
  'more-error': 'widgets.gallery.listingLoadMoreFailed',
  'stale-error': 'widgets.gallery.listingRefreshFailed',
};
const BOARD_NOTICES: Partial<Record<GalleryReadStatus, string>> = {
  error: 'widgets.gallery.boardsLoadFailed',
  'stale-error': 'widgets.gallery.boardsRefreshFailed',
};
const STARRED_NOTICES: Partial<Record<GalleryReadStatus, string>> = {
  error: 'widgets.gallery.starredLoadFailed',
  'stale-error': 'widgets.gallery.starredRefreshFailed',
};

/**
 * What the non-blocking notices say, for the widget's live region. A failed listing is left out: its error state is
 * an alert of its own.
 */
const getGalleryFailureAnnouncement = (
  {
    boardsState,
    listing,
    starredStrip,
  }: { boardsState: GalleryReadState; listing: GalleryListingState; starredStrip: GalleryStarredStrip },
  t: TFunction
): string =>
  [LISTING_NOTICES[listing.status], BOARD_NOTICES[boardsState.status], STARRED_NOTICES[starredStrip.state.status]]
    .filter((key) => key !== undefined)
    .map((key) => t(key))
    .join(' ');

const GalleryWidgetContent = ({
  actions,
  boardsState,
  filter,
  gallery,
  isWindowTruncated,
  listing,
  loadedItems,
  pinRevealIndex,
  projectName,
  region,
  runtime,
  setVisibleRange,
  sparseListing,
  starredStrip,
}: {
  actions: GalleryActions;
  boardsState: GalleryReadState;
  filter: GalleryItemsFilter;
  gallery: GalleryStateView;
  isWindowTruncated: boolean;
  listing: GalleryListingState;
  loadedItems: GalleryItem[];
  pinRevealIndex: GalleryData['pinRevealIndex'];
  projectName: string;
  region: GalleryWidgetProps['region'];
  runtime: GalleryWidgetRuntime;
  setVisibleRange: ((range: { endIndexExclusive: number; startIndex: number }) => void) | undefined;
  sparseListing: GalleryData['sparseListing'];
  starredStrip: GalleryStarredStrip;
}) => {
  const { t } = useTranslation();
  const itemActions = useGalleryItemActions();
  const contextValue = useMemo<GalleryWidgetContextValue>(
    () => ({
      actions,
      boardsState,
      filter,
      gallery,
      isWindowTruncated,
      itemActions,
      listing,
      loadedItems,
      pinRevealIndex,
      projectName,
      region,
      runtime,
      setVisibleRange,
      sparseListing,
      starredStrip,
    }),
    [
      actions,
      boardsState,
      filter,
      gallery,
      isWindowTruncated,
      itemActions,
      listing,
      loadedItems,
      pinRevealIndex,
      projectName,
      region,
      runtime,
      setVisibleRange,
      sparseListing,
      starredStrip,
    ]
  );

  return (
    <GalleryWidgetContext value={contextValue}>
      <GalleryBoardDragMonitor />
      <GalleryLayout region={region} />
      <GalleryAnnouncer message={getGalleryFailureAnnouncement({ boardsState, listing, starredStrip }, t)} />
    </GalleryWidgetContext>
  );
};

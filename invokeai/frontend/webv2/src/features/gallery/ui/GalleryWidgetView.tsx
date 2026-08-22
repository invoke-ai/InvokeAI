import type { GalleryItemsFilter } from '@features/gallery/data/queries';

import { getBoundedRecentImages } from '@features/gallery/core/recentImages';
import { getGallerySettings } from '@features/gallery/core/settings';
import { GALLERY_PAGE_SIZE, galleryItemNamesOptions } from '@features/gallery/data/queries';
import { StatusWidgetChip } from '@platform/ui';
import { useQueryClient } from '@tanstack/react-query';
import { ImageIcon } from 'lucide-react';
import { useCallback, useEffect, useEffectEvent, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryStateView } from './galleryStateView';

import { GalleryBoardDragMonitor } from './GalleryBoardDragMonitor';
import { GalleryLayout } from './GalleryLayout';
import {
  getGalleryPage,
  getGallerySearchTerm,
  getGallerySelectedBoardId,
  getGallerySemanticImageQuery,
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
import { GalleryWidgetContext, type GalleryActions, type GalleryWidgetContextValue } from './GalleryWidgetContext';
import { useGalleryActions } from './useGalleryActions';
import { useGalleryData } from './useGalleryData';

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

export const GalleryStatusChip = ({ count }: { count: number }) => {
  const { t } = useTranslation();

  return (
    <StatusWidgetChip icon={ImageIcon}>
      {t('widgets.gallery.statusChip', {
        count,
      })}
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
    liveFollowEnabled,
    liveProgressTarget,
    notifications,
    projectId,
    projectName,
    queueItems,
    ItemActionsProvider,
  } = useGalleryUi();
  const galleryView = getGalleryView(galleryValues);
  const searchTerm = getGallerySearchTerm(galleryValues);
  const recentImages = useMemo(() => getBoundedRecentImages(galleryValues.recentImages), [galleryValues.recentImages]);
  const page = getGalleryPage(galleryValues);
  const knownTotalImages = getGalleryTotalImages(galleryValues);
  const settings = getGallerySettings(galleryValues);
  const queryClient = useQueryClient();
  const semanticQuery = useGallerySemanticImageQuery(galleryValues.semanticImageQuery);
  const data = useGalleryData({
    galleryView,
    page,
    recentImages,
    searchTerm,
    selectedBoardId: getGallerySelectedBoardId(galleryValues, []),
    semanticQuery,
    settings,
  });

  // A failed similarity search renders exactly like an empty one ("no images
  // match"), so the failure must reach the user some other way. Scoped to
  // semantic mode: the board listing kept its pre-existing quiet behavior.
  const semanticError = semanticQuery ? data.queryError : null;

  useEffect(() => {
    if (!semanticError) {
      return;
    }

    notifications.reportError({
      area: 'gallery-semantic-search',
      message: semanticError.message,
      namespace: 'gallery',
    });
  }, [notifications, semanticError]);

  const { loadMore, total } = data;
  const selectedBoardId = getGallerySelectedBoardId(galleryValues, data.boards);
  const gallery = getGalleryStateView(
    galleryValues,
    data.boards,
    data.items,
    data.isLoadingItems,
    queueItems,
    liveFollowEnabled,
    liveProgressTarget
  );
  const lastPublishedTotalRef = useRef<number | null>(null);
  const itemActionFilterIdentity = useMemo(() => JSON.stringify(data.filter), [data.filter]);
  const loadOrderedItemRefs = useCallback(
    async (signal: AbortSignal) => {
      signal.throwIfAborted();
      const result = await queryClient.fetchQuery(galleryItemNamesOptions(data.filter));

      signal.throwIfAborted();
      return result.items;
    },
    [data.filter, queryClient]
  );
  const itemActionContextRef = useRef<GalleryItemActionContext | null>(null);
  const galleryLocationRef = useRef({ galleryView, selectedBoardId });

  // This ref is a live read port for an in-flight deletion. An effect would
  // leave a commit-sized stale window, while the action must compare against
  // the exact filter and selection from the latest render.
  // eslint-disable-next-line react/react-compiler
  itemActionContextRef.current = {
    filterIdentity: itemActionFilterIdentity,
    items: gallery.items,
    loadOrderedRefs: loadOrderedItemRefs,
    selectedItemKey: gallery.selectedItemKey,
  };
  // This is the matching live read port for in-flight uploads. The upload
  // target is captured at launch, while completion visibility must use the
  // board and view from the latest render.
  // eslint-disable-next-line react/react-compiler
  galleryLocationRef.current = { galleryView, selectedBoardId };

  const getItemActionContext = useCallback(() => itemActionContextRef.current, []);
  const getCurrentGalleryLocation = useCallback(() => galleryLocationRef.current, []);

  const actions = useGalleryActions({
    boards: data.boards,
    getCurrentGalleryLocation,
    loadMore,
    selectedBoardId,
  });

  // Publish the backend total into widget values so the manifest footer can
  // render page navigation without its own fetch, and clamp the page when the
  // query shrinks (e.g. after deletions).
  //
  // The effect reacts only to this instance's fetched `total` — the published
  // state (`knownTotalImages`) is read through a ref on purpose. Several
  // gallery views can be mounted at once (e.g. the bottom status chip plus an
  // expanded gallery), and if each instance re-published whenever the shared
  // state disagreed with its own in-flight total, two instances mid-refetch
  // would dispatch in a loop until React aborts with "maximum update depth".
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

  useEffect(() => {
    if (settings.paginationMode !== 'paginated' || total === null) {
      return;
    }

    const maxPage = Math.max(0, Math.ceil(total / GALLERY_PAGE_SIZE) - 1);

    if (page > maxPage) {
      galleryCommands.setPage(maxPage);
    }
  }, [galleryCommands, page, settings.paginationMode, total]);

  if (region === 'bottom' && presentation !== 'expanded') {
    return <GalleryStatusChip count={total ?? gallery.items.length} />;
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
        filter={data.filter}
        gallery={gallery}
        isWindowTruncated={data.isWindowTruncated}
        projectName={projectName}
        region={region}
        runtime={runtime}
      />
    </ItemActionsProvider>
  );
};

const GalleryWidgetContent = ({
  actions,
  filter,
  gallery,
  isWindowTruncated,
  projectName,
  region,
  runtime,
}: {
  actions: GalleryActions;
  filter: GalleryItemsFilter;
  gallery: GalleryStateView;
  isWindowTruncated: boolean;
  projectName: string;
  region: GalleryWidgetProps['region'];
  runtime: GalleryWidgetRuntime;
}) => {
  const itemActions = useGalleryItemActions();
  const contextValue = useMemo<GalleryWidgetContextValue>(
    () => ({ actions, filter, gallery, isWindowTruncated, itemActions, projectName, region, runtime }),
    [actions, filter, gallery, isWindowTruncated, itemActions, projectName, region, runtime]
  );

  return (
    <GalleryWidgetContext value={contextValue}>
      <GalleryBoardDragMonitor />
      <GalleryLayout region={region} />
    </GalleryWidgetContext>
  );
};

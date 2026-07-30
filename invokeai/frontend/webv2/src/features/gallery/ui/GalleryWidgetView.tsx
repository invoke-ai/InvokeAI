import { getBoundedRecentImages } from '@features/gallery/core/recentImages';
import { getGallerySettings } from '@features/gallery/core/settings';
import { GALLERY_PAGE_SIZE, galleryItemNamesOptions } from '@features/gallery/data/queries';
import { StatusWidgetChip } from '@platform/ui';
import { useQueryClient } from '@tanstack/react-query';
import { ImageIcon } from 'lucide-react';
import { useCallback, useEffect, useEffectEvent, useMemo, useRef } from 'react';

import type { GalleryStateView } from './galleryStateView';

import { GalleryPanelContent } from './GalleryPanelContent';
import {
  getGalleryPage,
  getGalleryProjectBoardId,
  getGallerySearchTerm,
  getGallerySelectedBoardId,
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

export const GalleryWidgetView = ({ presentation, region, runtime }: GalleryWidgetProps) => {
  const {
    gallery: galleryCommands,
    galleryValues,
    generateValues,
    liveFollowEnabled,
    liveProgressTarget,
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
  const data = useGalleryData({
    galleryView,
    page,
    recentImages,
    searchTerm,
    selectedBoardId: getGallerySelectedBoardId(galleryValues, []),
    settings,
  });

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

  const getItemActionContext = useCallback(() => itemActionContextRef.current, []);

  const actions = useGalleryActions({
    boards: data.boards,
    loadMore,
    projectBoardId: getGalleryProjectBoardId(galleryValues),
    projectName,
    selectedBoardId,
  });

  const isWidePlacement = region === 'center' || (region === 'bottom' && presentation === 'expanded');

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
    return <StatusWidgetChip icon={ImageIcon}>Gallery: {total ?? gallery.items.length}</StatusWidgetChip>;
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
        gallery={gallery}
        isWindowTruncated={data.isWindowTruncated}
        layout={isWidePlacement ? 'wide' : 'stacked'}
        projectName={projectName}
        runtime={runtime}
      />
    </ItemActionsProvider>
  );
};

const GalleryWidgetContent = ({
  actions,
  gallery,
  isWindowTruncated,
  layout,
  projectName,
  runtime,
}: {
  actions: GalleryActions;
  gallery: GalleryStateView;
  isWindowTruncated: boolean;
  layout: 'stacked' | 'wide';
  projectName: string;
  runtime: GalleryWidgetRuntime;
}) => {
  const itemActions = useGalleryItemActions();
  const contextValue = useMemo<GalleryWidgetContextValue>(
    () => ({ actions, gallery, isWindowTruncated, itemActions, projectName, runtime }),
    [actions, gallery, isWindowTruncated, itemActions, projectName, runtime]
  );

  return (
    <GalleryWidgetContext value={contextValue}>
      <GalleryPanelContent layout={layout} />
    </GalleryWidgetContext>
  );
};

import { getGallerySettings } from '@features/gallery/core/settings';
import { StatusWidgetChip } from '@platform/ui';
import { ImageIcon } from 'lucide-react';
import { useCallback, useEffect, useEffectEvent, useMemo, useRef } from 'react';

import type { GalleryStateView } from './galleryStateView';

import { GalleryPanelContent } from './GalleryPanelContent';
import {
  getGalleryPage,
  getGalleryProjectBoardId,
  getGalleryImagesRefreshToken,
  getGalleryRecentImagesKey,
  getGalleryRefreshToken,
  getGallerySearchTerm,
  getGallerySelectedBoardId,
  getGalleryStateView,
  getGalleryTotalImages,
  getGalleryView,
} from './galleryStateView';
import {
  useGalleryImageActions,
  useGalleryUi,
  type GalleryWidgetProps,
  type GalleryWidgetRuntime,
} from './GalleryUiContext';
import { GalleryWidgetContext, type GalleryActions, type GalleryWidgetContextValue } from './GalleryWidgetContext';
import { useGalleryActions } from './useGalleryActions';
import { GALLERY_PAGE_SIZE, useGalleryData } from './useGalleryData';

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
    ImageActionsProvider,
  } = useGalleryUi();
  const galleryView = getGalleryView(galleryValues);
  const searchTerm = getGallerySearchTerm(galleryValues);
  const recentImagesKey = getGalleryRecentImagesKey(galleryValues);
  const refreshToken = getGalleryRefreshToken(galleryValues);
  const imageRefreshToken = getGalleryImagesRefreshToken(galleryValues);
  const page = getGalleryPage(galleryValues);
  const knownTotalImages = getGalleryTotalImages(galleryValues);
  const settings = getGallerySettings(galleryValues);
  const data = useGalleryData({
    galleryView,
    imageRefreshToken,
    page,
    recentImagesKey,
    refreshToken,
    searchTerm,
    selectedBoardId: getGallerySelectedBoardId(galleryValues, []),
    settings,
  });

  const { loadMore, patchImages, total } = data;
  const selectedBoardId = getGallerySelectedBoardId(galleryValues, data.boards);
  const gallery = getGalleryStateView(
    galleryValues,
    data.boards,
    data.images,
    data.isLoadingImages,
    queueItems,
    liveFollowEnabled,
    liveProgressTarget
  );

  const onStarredChange = useCallback(
    (imageNames: string[], starred: boolean) => {
      patchImages((images) =>
        images.map((image) => (imageNames.includes(image.imageName) ? { ...image, starred } : image))
      );
    },
    [patchImages]
  );

  const lastPublishedTotalRef = useRef<number | null>(null);

  // After a deletion that takes out the previewed image, move the selection to
  // the image that now occupies the old index, else the one before it.
  const onImagesDeleted = useCallback(
    (imageNames: string[]) => {
      const deletedNames = new Set(imageNames);
      const images = gallery.images;
      const anchorName = gallery.selectedImageName;

      if (!anchorName || !deletedNames.has(anchorName)) {
        return;
      }

      const anchorIndex = images.findIndex((image) => image.imageName === anchorName);

      if (anchorIndex === -1) {
        return;
      }

      const remaining = images.filter((image) => !deletedNames.has(image.imageName));
      const remainingBeforeAnchor = images
        .slice(0, anchorIndex)
        .filter((image) => !deletedNames.has(image.imageName)).length;
      const nextImage = remaining[remainingBeforeAnchor] ?? remaining[remainingBeforeAnchor - 1] ?? null;

      if (nextImage) {
        galleryCommands.selectImage(nextImage);
      }
    },
    [gallery.images, gallery.selectedImageName, galleryCommands]
  );

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
    return <StatusWidgetChip icon={ImageIcon}>Gallery: {total ?? gallery.images.length}</StatusWidgetChip>;
  }

  return (
    <ImageActionsProvider
      boards={data.boards}
      generateValues={generateValues}
      projectId={projectId}
      onImagesDeleted={onImagesDeleted}
      onStarredChange={onStarredChange}
    >
      <GalleryWidgetContent
        actions={actions}
        gallery={gallery}
        layout={isWidePlacement ? 'wide' : 'stacked'}
        projectName={projectName}
        runtime={runtime}
      />
    </ImageActionsProvider>
  );
};

const GalleryWidgetContent = ({
  actions,
  gallery,
  layout,
  projectName,
  runtime,
}: {
  actions: GalleryActions;
  gallery: GalleryStateView;
  layout: 'stacked' | 'wide';
  projectName: string;
  runtime: GalleryWidgetRuntime;
}) => {
  const imageActions = useGalleryImageActions();
  const contextValue = useMemo<GalleryWidgetContextValue>(
    () => ({ actions, gallery, imageActions, projectName, runtime }),
    [actions, gallery, imageActions, projectName, runtime]
  );

  return (
    <GalleryWidgetContext value={contextValue}>
      <GalleryPanelContent layout={layout} />
    </GalleryWidgetContext>
  );
};

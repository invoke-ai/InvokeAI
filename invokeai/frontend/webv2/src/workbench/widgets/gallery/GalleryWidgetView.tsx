import { useCallback, useEffect, useMemo, useRef } from 'react';
import { ImageIcon } from 'lucide-react';

import { StatusWidgetChip } from '../../components/WidgetFrames';
import { useImageActions } from '../../components/useImageActions';
import { getGallerySettings } from '../../gallery/settings';
import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';
import { GalleryPanelContent } from './GalleryPanelContent';
import {
  getGalleryPage,
  getGalleryProjectBoardId,
  getGalleryRecentImagesKey,
  getGalleryRefreshToken,
  getGallerySearchTerm,
  getGallerySelectedBoardId,
  getGalleryStateView,
  getGalleryTotalImages,
  getGalleryView,
} from './galleryStateView';
import { GalleryWidgetContext, type GalleryWidgetContextValue } from './GalleryWidgetContext';
import { useGalleryActions } from './useGalleryActions';
import { GALLERY_PAGE_SIZE, useGalleryData } from './useGalleryData';

export const GalleryWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const { activeProject, dispatch } = useWorkbench();
  const galleryValues = activeProject.widgetStates.gallery.values;
  const galleryView = getGalleryView(galleryValues);
  const searchTerm = getGallerySearchTerm(galleryValues);
  const recentImagesKey = getGalleryRecentImagesKey(galleryValues);
  const refreshToken = getGalleryRefreshToken(galleryValues);
  const page = getGalleryPage(galleryValues);
  const knownTotalImages = getGalleryTotalImages(galleryValues);
  const settings = getGallerySettings(galleryValues);
  const onError = useCallback((message: string) => dispatch({ message, type: 'recordError' }), [dispatch]);
  const data = useGalleryData({
    galleryView,
    onError,
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
    activeProject.queue.items
  );
  const onStarredChange = useCallback(
    (imageNames: string[], starred: boolean) => {
      patchImages((images) =>
        images.map((image) => (imageNames.includes(image.imageName) ? { ...image, starred } : image))
      );
    },
    [patchImages]
  );
  const galleryImagesRef = useRef(gallery.images);
  const selectedImageNameRef = useRef(gallery.selectedImageName);

  galleryImagesRef.current = gallery.images;
  selectedImageNameRef.current = gallery.selectedImageName;

  // After a deletion that takes out the previewed image, move the selection to
  // the image that now occupies the old index, else the one before it.
  const onImagesDeleted = useCallback(
    (imageNames: string[]) => {
      const deletedNames = new Set(imageNames);
      const images = galleryImagesRef.current;
      const anchorName = selectedImageNameRef.current;

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
        dispatch({ image: nextImage, type: 'selectGalleryImage' });
      }
    },
    [dispatch]
  );
  const imageActions = useImageActions({ boards: data.boards, dispatch, onImagesDeleted, onStarredChange });
  const actions = useGalleryActions({
    boards: data.boards,
    dispatch,
    loadMore,
    projectBoardId: getGalleryProjectBoardId(galleryValues),
    projectName: activeProject.name,
    selectedBoardId,
  });
  const contextValue = useMemo<GalleryWidgetContextValue>(
    () => ({ actions, gallery, imageActions, projectName: activeProject.name }),
    [actions, activeProject.name, gallery, imageActions]
  );
  const isWidePlacement = region === 'center' || (region === 'bottom' && presentation === 'expanded');

  // Publish the backend total into widget values so the manifest footer can
  // render page navigation without its own fetch, and clamp the page when the
  // query shrinks (e.g. after deletions).
  useEffect(() => {
    if (total !== null && total !== knownTotalImages) {
      dispatch({ totalImages: total, type: 'setGalleryPageInfo' });
    }
  }, [dispatch, knownTotalImages, total]);

  useEffect(() => {
    if (settings.paginationMode !== 'paginated' || total === null) {
      return;
    }

    const maxPage = Math.max(0, Math.ceil(total / GALLERY_PAGE_SIZE) - 1);

    if (page > maxPage) {
      dispatch({ page: maxPage, type: 'setGalleryPage' });
    }
  }, [dispatch, page, settings.paginationMode, total]);

  if (region === 'bottom' && presentation !== 'expanded') {
    return <StatusWidgetChip icon={ImageIcon}>Gallery: {total ?? gallery.images.length}</StatusWidgetChip>;
  }

  return (
    <GalleryWidgetContext value={contextValue}>
      <GalleryPanelContent layout={isWidePlacement ? 'wide' : 'stacked'} />
    </GalleryWidgetContext>
  );
};

/* eslint-disable react/react-compiler */
import type { GeneratedImageContract, WidgetViewProps } from '@workbench/types';

import { getGallerySettings, type GallerySettings } from '@workbench/gallery/settings';
import { useImageActions } from '@workbench/image-actions';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { createGenerateFormValuesSelector } from '@workbench/widgets/generate/generateFormViewModel';
import {
  useActiveProjectId,
  useActiveProjectName,
  useActiveProjectSelector,
  useWidgetValuesSelector,
  useWorkbenchDispatch,
} from '@workbench/WorkbenchContext';
import { areArraysEqual, createStableSelector } from '@workbench/workbenchSelectors';
import { ImageIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef } from 'react';

import { GalleryPanelContent } from './GalleryPanelContent';
import {
  getGalleryCompareImage,
  getGalleryPage,
  getGalleryProjectBoardId,
  getGalleryImagesRefreshToken,
  getGalleryRecentImagesKey,
  getGalleryRefreshToken,
  getGallerySearchTerm,
  getGallerySelectedBoardId,
  getGallerySelectedImageNames,
  getGalleryStateView,
  getGalleryTotalImages,
  getGalleryView,
} from './galleryStateView';
import { GalleryWidgetContext, type GalleryWidgetContextValue } from './GalleryWidgetContext';
import { useGalleryActions } from './useGalleryActions';
import { GALLERY_PAGE_SIZE, useGalleryData } from './useGalleryData';

interface GalleryWidgetSelectedValues extends Record<string, unknown>, GallerySettings {
  compareImage: GeneratedImageContract | null;
  galleryImagesRefreshToken: string;
  galleryPage: number;
  galleryRefreshToken: string;
  galleryTotalImages: number | null;
  galleryView: ReturnType<typeof getGalleryView>;
  projectBoardId: string | null;
  recentImages: GeneratedImageContract[] | undefined;
  searchTerm: string;
  selectedBoardId: string;
  selectedImageName: string | null;
  selectedImageNames: string[];
}

const areGallerySettingsEqual = (left: GallerySettings, right: GallerySettings): boolean =>
  left.boardOrderBy === right.boardOrderBy &&
  left.boardOrderDir === right.boardOrderDir &&
  left.imageDensityPercent === right.imageDensityPercent &&
  left.imageOrderDir === right.imageOrderDir &&
  left.paginationMode === right.paginationMode &&
  left.showArchivedBoards === right.showArchivedBoards &&
  left.showDateBoards === right.showDateBoards &&
  left.showImageDimensions === right.showImageDimensions &&
  left.starredFirst === right.starredFirst &&
  left.thumbnailFit === right.thumbnailFit;

const areGalleryWidgetSelectedValuesEqual = (
  left: GalleryWidgetSelectedValues,
  right: GalleryWidgetSelectedValues
): boolean =>
  left.compareImage === right.compareImage &&
  left.galleryImagesRefreshToken === right.galleryImagesRefreshToken &&
  left.galleryPage === right.galleryPage &&
  left.galleryRefreshToken === right.galleryRefreshToken &&
  left.galleryTotalImages === right.galleryTotalImages &&
  left.galleryView === right.galleryView &&
  left.projectBoardId === right.projectBoardId &&
  left.recentImages === right.recentImages &&
  left.searchTerm === right.searchTerm &&
  left.selectedBoardId === right.selectedBoardId &&
  left.selectedImageName === right.selectedImageName &&
  areArraysEqual(left.selectedImageNames, right.selectedImageNames) &&
  areGallerySettingsEqual(left, right);

const selectGalleryWidgetValues = createStableSelector(
  (values: Record<string, unknown>): GalleryWidgetSelectedValues => {
    const settings = getGallerySettings(values);

    return {
      ...settings,
      compareImage: getGalleryCompareImage(values),
      galleryImagesRefreshToken: getGalleryImagesRefreshToken(values),
      galleryPage: getGalleryPage(values),
      galleryRefreshToken: getGalleryRefreshToken(values),
      galleryTotalImages: getGalleryTotalImages(values),
      galleryView: getGalleryView(values),
      projectBoardId: getGalleryProjectBoardId(values),
      recentImages: Array.isArray(values.recentImages) ? (values.recentImages as GeneratedImageContract[]) : undefined,
      searchTerm: getGallerySearchTerm(values),
      selectedBoardId: getGallerySelectedBoardId(values, []),
      selectedImageName: typeof values.selectedImageName === 'string' ? values.selectedImageName : null,
      selectedImageNames: getGallerySelectedImageNames(values),
    };
  },
  areGalleryWidgetSelectedValuesEqual
);

const selectGenerateRecallValues = createGenerateFormValuesSelector();

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

export const GalleryWidgetView = ({ presentation, region, runtime }: WidgetViewProps) => {
  const dispatch = useWorkbenchDispatch();
  const projectId = useActiveProjectId();
  const galleryValues = useWidgetValuesSelector('gallery', selectGalleryWidgetValues);
  const generateValues = useWidgetValuesSelector('generate', selectGenerateRecallValues);
  const projectName = useActiveProjectName();
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const galleryView = getGalleryView(galleryValues);
  const searchTerm = getGallerySearchTerm(galleryValues);
  const recentImagesKey = getGalleryRecentImagesKey(galleryValues);
  const refreshToken = getGalleryRefreshToken(galleryValues);
  const imageRefreshToken = getGalleryImagesRefreshToken(galleryValues);
  const page = getGalleryPage(galleryValues);
  const knownTotalImages = getGalleryTotalImages(galleryValues);
  const settings = getGallerySettings(galleryValues);
  const onError = useCallback(
    (message: string) =>
      dispatch({ area: 'gallery-data', message, namespace: 'gallery', projectId, type: 'recordError' }),
    [dispatch, projectId]
  );

  const data = useGalleryData({
    galleryView,
    imageRefreshToken,
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
  const gallery = getGalleryStateView(galleryValues, data.boards, data.images, data.isLoadingImages, queueItems);

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
  const lastPublishedTotalRef = useRef<number | null>(null);

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

  const imageActions = useImageActions({
    boards: data.boards,
    dispatch,
    generateValues,
    onImagesDeleted,
    onStarredChange,
    projectId,
  });

  const actions = useGalleryActions({
    boards: data.boards,
    dispatch,
    loadMore,
    projectBoardId: getGalleryProjectBoardId(galleryValues),
    projectName,
    selectedBoardId,
  });

  const contextValue = useMemo<GalleryWidgetContextValue>(
    () => ({ actions, gallery, imageActions, projectName, runtime }),
    [actions, gallery, imageActions, projectName, runtime]
  );

  const isWidePlacement = region === 'center' || (region === 'bottom' && presentation === 'expanded');

  // Publish the backend total into widget values so the manifest footer can
  // render page navigation without its own fetch, and clamp the page when the
  // query shrinks (e.g. after deletions).
  useEffect(() => {
    if (total === knownTotalImages) {
      lastPublishedTotalRef.current = null;
      return;
    }

    if (typeof total !== 'number' || !Number.isFinite(total)) {
      return;
    }

    if (
      shouldPublishGalleryTotal({
        knownTotalImages,
        lastPublishedTotal: lastPublishedTotalRef.current,
        total,
      })
    ) {
      lastPublishedTotalRef.current = total;
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

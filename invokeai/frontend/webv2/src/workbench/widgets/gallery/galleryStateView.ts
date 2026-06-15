import type { GalleryBoard, GalleryImage, GalleryView } from '@workbench/gallery/api';
import type { GeneratedImageContract, QueueItem } from '@workbench/types';

import { getGallerySettings, type GallerySettings } from '@workbench/gallery/settings';
import { sanitizeBatchCount } from '@workbench/generation/batch';

const UNCATEGORIZED_BOARD: GalleryBoard = {
  archived: false,
  assetCount: 0,
  id: 'none',
  imageCount: 0,
  kind: 'uncategorized',
  name: 'Uncategorized',
};

/**
 * A grid cell standing in for an image that an in-flight queue item will
 * deliver to the currently viewed board once generation completes.
 */
export interface GalleryQueuePlaceholder {
  id: string;
  queueItemId: string;
  itemIndex: number;
  width: number;
  height: number;
}

export interface GalleryStateView {
  boards: GalleryBoard[];
  compareImageName: string | null;
  galleryView: GalleryView;
  images: GalleryImage[];
  isLoading: boolean;
  pendingPlaceholders: GalleryQueuePlaceholder[];
  projectBoardId: string | null;
  searchTerm: string;
  selectedBoardId: string;
  selectedImageName: string | null;
  selectedImageNames: string[];
  settings: GallerySettings;
}

export const getGalleryView = (values: Record<string, unknown>): GalleryView =>
  values.galleryView === 'assets' ? 'assets' : 'images';

export const getGallerySearchTerm = (values: Record<string, unknown>): string =>
  typeof values.searchTerm === 'string' ? values.searchTerm : '';

export const getGallerySelectedBoardId = (values: Record<string, unknown>, backendBoards: GalleryBoard[]): string => {
  const selectedBoardId = typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none';

  if (backendBoards.length === 0 || backendBoards.some((board) => board.id === selectedBoardId)) {
    return selectedBoardId;
  }

  return 'none';
};

export const getGallerySelectedImageNames = (values: Record<string, unknown>): string[] => {
  if (Array.isArray(values.selectedImageNames)) {
    return (values.selectedImageNames as unknown[]).filter((name): name is string => typeof name === 'string');
  }

  return typeof values.selectedImageName === 'string' ? [values.selectedImageName] : [];
};

export const getGalleryPage = (values: Record<string, unknown>): number =>
  typeof values.galleryPage === 'number' && Number.isFinite(values.galleryPage)
    ? Math.max(0, Math.floor(values.galleryPage))
    : 0;

export const getGalleryTotalImages = (values: Record<string, unknown>): number | null =>
  typeof values.galleryTotalImages === 'number' && Number.isFinite(values.galleryTotalImages)
    ? Math.max(0, values.galleryTotalImages)
    : null;

export const getGalleryRefreshToken = (values: Record<string, unknown>): string =>
  typeof values.galleryRefreshToken === 'string' ? values.galleryRefreshToken : '';

export const getGalleryProjectBoardId = (values: Record<string, unknown>): string | null =>
  typeof values.projectBoardId === 'string' ? values.projectBoardId : null;

export const getGalleryCompareImage = (values: Record<string, unknown>): GeneratedImageContract | null => {
  const compareImage = values.compareImage;

  if (
    compareImage &&
    typeof compareImage === 'object' &&
    typeof (compareImage as GeneratedImageContract).imageName === 'string'
  ) {
    return compareImage as GeneratedImageContract;
  }

  return null;
};

export const getGalleryRecentImagesKey = (values: Record<string, unknown>): string => {
  if (!Array.isArray(values.recentImages)) {
    return '';
  }

  return (values.recentImages as GeneratedImageContract[]).map((image) => image.imageName).join('\0');
};

const getFiniteNumber = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

export const getGalleryQueuePlaceholders = (
  queueItems: QueueItem[],
  {
    galleryView,
    searchTerm,
    selectedBoardId,
  }: { galleryView: GalleryView; searchTerm: string; selectedBoardId: string }
): GalleryQueuePlaceholder[] => {
  if (galleryView !== 'images' || searchTerm.trim() !== '') {
    return [];
  }

  const placeholders: GalleryQueuePlaceholder[] = [];

  for (const item of queueItems) {
    if (item.status !== 'pending' && item.status !== 'running') {
      continue;
    }

    if (item.snapshot.destination !== 'gallery') {
      continue;
    }

    const galleryValues = item.snapshot.widgetStates.gallery?.values ?? {};
    const targetBoardId = typeof galleryValues.selectedBoardId === 'string' ? galleryValues.selectedBoardId : 'none';

    if (targetBoardId !== selectedBoardId) {
      continue;
    }

    const generateValues = item.snapshot.widgetStates.generate?.values ?? {};
    const expectedImageCount = item.backendItemIds?.length
      ? item.backendItemIds.length
      : sanitizeBatchCount(generateValues.batchCount);
    const width = getFiniteNumber(generateValues.width, 1024);
    const height = getFiniteNumber(generateValues.height, 1024);
    const completedBackendItemIds = new Set(item.completedBackendItemIds ?? []);
    const cancelledBackendItemIds = new Set(item.cancelledBackendItemIds ?? []);

    if (item.backendItemIds?.length) {
      for (let index = 0; index < item.backendItemIds.length; index += 1) {
        const backendItemId = item.backendItemIds[index];

        if (
          backendItemId === undefined ||
          completedBackendItemIds.has(backendItemId) ||
          cancelledBackendItemIds.has(backendItemId)
        ) {
          continue;
        }

        placeholders.push({ height, id: `${item.id}:${index}`, itemIndex: index + 1, queueItemId: item.id, width });
      }

      continue;
    }

    for (let index = 0; index < expectedImageCount; index++) {
      placeholders.push({ height, id: `${item.id}:${index}`, itemIndex: index + 1, queueItemId: item.id, width });
    }
  }

  return placeholders;
};

export const getGalleryStateView = (
  values: Record<string, unknown>,
  backendBoards: GalleryBoard[],
  backendImages: GalleryImage[] | null,
  isLoading: boolean,
  queueItems: QueueItem[] = []
): GalleryStateView => {
  const localImages = Array.isArray(values.recentImages)
    ? (values.recentImages as GeneratedImageContract[]).map((image) => ({
        ...image,
        boardId: 'none',
        imageCategory: 'general' as const,
        starred: false,
      }))
    : [];
  const images = backendImages ?? (isLoading ? [] : localImages);
  const selectedImageName = typeof values.selectedImageName === 'string' ? values.selectedImageName : null;
  const galleryView = getGalleryView(values);
  const settings = getGallerySettings(values);
  const searchTerm = getGallerySearchTerm(values);
  const boards = backendBoards.length ? backendBoards : [{ ...UNCATEGORIZED_BOARD, imageCount: images.length }];
  const selectedBoardId = getGallerySelectedBoardId(values, backendBoards);
  const compareImage = getGalleryCompareImage(values);

  return {
    boards,
    compareImageName: compareImage?.imageName ?? null,
    galleryView,
    images,
    isLoading,
    pendingPlaceholders: getGalleryQueuePlaceholders(queueItems, { galleryView, searchTerm, selectedBoardId }),
    projectBoardId: getGalleryProjectBoardId(values),
    searchTerm,
    selectedBoardId,
    selectedImageName: images.some((image) => image.imageName === selectedImageName) ? selectedImageName : null,
    selectedImageNames: getGallerySelectedImageNames(values),
    settings,
  };
};

export const getBoardCounts = (board: GalleryBoard): { assetCount: number; imageCount: number } => ({
  assetCount: board.assetCount,
  imageCount: board.imageCount,
});

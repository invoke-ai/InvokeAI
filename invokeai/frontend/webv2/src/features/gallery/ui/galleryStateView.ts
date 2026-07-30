import type {
  GalleryBoard,
  GalleryImage,
  GalleryOrderDir,
  GalleryView,
  GeneratedImageContract,
} from '@features/gallery/core/types';
import type { QueueItem } from '@features/queue/contracts';

import { normalizeGalleryImage } from '@features/gallery/core/image';
import { getBoundedRecentImages } from '@features/gallery/core/recentImages';
import { getGallerySettings, type GallerySettings } from '@features/gallery/core/settings';
import { getQueueItemSnapshotBatchCount, getQueueItemSnapshotDimensions } from '@features/queue/contracts';

const UNCATEGORIZED_BOARD: GalleryBoard = {
  archived: false,
  assetCount: 0,
  id: 'none',
  imageCount: 0,
  kind: 'uncategorized',
  name: 'Uncategorized',
  videoCount: 0,
};

/**
 * A grid cell standing in for an image that an in-flight queue item will
 * deliver to the currently viewed board once generation completes.
 */
export interface GalleryQueuePlaceholder {
  boardId: string;
  id: string;
  queueItemId: string;
  itemIndex: number;
  width: number;
  height: number;
  /**
   * The backend queue item id, once the slot has been claimed by one — null while the
   * submission is still local-only. Progress keyed by *local* id cannot distinguish
   * two slots of the same batch running on two GPUs, so surfaces that need per-session
   * facts (which GPU is rendering this tile) resolve them through this id.
   */
  backendItemId: number | null;
}

export interface GalleryLiveTarget {
  queueItemId: string;
  itemIndex: number;
}

export interface GalleryGenerationSequence {
  chronologicalSlots: GalleryQueuePlaceholder[];
  liveSlot: GalleryQueuePlaceholder | null;
}

export type GalleryCurrentItem =
  | { kind: 'image'; imageName: string }
  | { kind: 'placeholder'; placeholder: GalleryQueuePlaceholder }
  | null;

export interface GalleryStateView {
  boards: GalleryBoard[];
  compareImageName: string | null;
  currentItem: GalleryCurrentItem;
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

interface SortableGalleryQueueSlot {
  backendItemId: number | null;
  placeholder: GalleryQueuePlaceholder;
  submittedAt: string;
}

interface GalleryOrderImage {
  starred?: boolean;
}

export const getGalleryPlaceholderInsertionIndex = (
  images: GalleryOrderImage[],
  imageOrderDir: GalleryOrderDir,
  starredFirst: boolean
): number => {
  if (imageOrderDir !== 'DESC') {
    return images.length;
  }

  if (!starredFirst) {
    return 0;
  }

  const firstUnstarredIndex = images.findIndex((image) => !image.starred);

  return firstUnstarredIndex === -1 ? images.length : firstUnstarredIndex;
};

export const getGalleryGenerationSequence = (
  queueItems: QueueItem[],
  liveTarget: GalleryLiveTarget | null
): GalleryGenerationSequence => {
  const sortableSlots: SortableGalleryQueueSlot[] = [];

  for (const item of queueItems) {
    if ((item.status !== 'pending' && item.status !== 'running') || item.snapshot.destination !== 'gallery') {
      continue;
    }

    const boardId = item.snapshot.galleryBoardId ?? 'none';
    const { width, height } = getQueueItemSnapshotDimensions(item, { height: 1024, width: 1024 });
    const completedBackendItemIds = new Set(item.completedBackendItemIds ?? []);
    const cancelledBackendItemIds = new Set(item.cancelledBackendItemIds ?? []);
    const slotCount = item.backendItemIds?.length ?? getQueueItemSnapshotBatchCount(item);

    for (let index = 0; index < slotCount; index += 1) {
      const backendItemId = item.backendItemIds?.[index] ?? null;

      if (
        backendItemId !== null &&
        (completedBackendItemIds.has(backendItemId) || cancelledBackendItemIds.has(backendItemId))
      ) {
        continue;
      }

      sortableSlots.push({
        backendItemId,
        placeholder: {
          backendItemId,
          boardId,
          height,
          id: `${item.id}:${index}`,
          itemIndex: index + 1,
          queueItemId: item.id,
          width,
        },
        submittedAt: item.snapshot.submittedAt,
      });
    }
  }

  sortableSlots.sort((a, b) => {
    if (a.backendItemId !== null && b.backendItemId !== null) {
      return a.backendItemId - b.backendItemId;
    }

    if (a.backendItemId !== null) {
      return -1;
    }

    if (b.backendItemId !== null) {
      return 1;
    }

    return (
      a.submittedAt.localeCompare(b.submittedAt) ||
      a.placeholder.queueItemId.localeCompare(b.placeholder.queueItemId) ||
      a.placeholder.itemIndex - b.placeholder.itemIndex
    );
  });

  const chronologicalSlots = sortableSlots.map(({ placeholder }) => placeholder);
  const liveSlot = liveTarget
    ? (chronologicalSlots.find(
        (slot) => slot.queueItemId === liveTarget.queueItemId && slot.itemIndex === liveTarget.itemIndex
      ) ?? null)
    : null;

  return { chronologicalSlots, liveSlot };
};

/**
 * The slots for every concurrently-running session, in chronological order.
 *
 * Multi-GPU runs one session per GPU, so more than one slot can be live at once.
 * Ordering comes from `chronologicalSlots` (sorted by backend item id) rather than
 * from the order the targets happened to start reporting, so tiles keep a stable
 * left-to-right position for as long as they run.
 */
export const getGalleryLiveSlots = (
  chronologicalSlots: GalleryQueuePlaceholder[],
  liveTargets: readonly GalleryLiveTarget[]
): GalleryQueuePlaceholder[] =>
  liveTargets.length === 0
    ? []
    : chronologicalSlots.filter((slot) =>
        liveTargets.some((target) => target.queueItemId === slot.queueItemId && target.itemIndex === slot.itemIndex)
      );

export const getGalleryCurrentItem = ({
  activePlaceholder,
  isComparisonActive,
  liveFollowEnabled,
  selectedImageName,
}: {
  activePlaceholder: GalleryQueuePlaceholder | null;
  isComparisonActive: boolean;
  liveFollowEnabled: boolean;
  selectedImageName: string | null;
}): GalleryCurrentItem => {
  if (liveFollowEnabled && !isComparisonActive && activePlaceholder) {
    return { kind: 'placeholder', placeholder: activePlaceholder };
  }

  return selectedImageName ? { imageName: selectedImageName, kind: 'image' } : null;
};

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

export const getGallerySelectedImagePage = (values: Record<string, unknown>): number =>
  typeof values.selectedImagePage === 'number' && Number.isFinite(values.selectedImagePage)
    ? Math.max(0, Math.floor(values.selectedImagePage))
    : getGalleryPage(values);

export interface GallerySelectedImageQuery {
  boardId: string;
  galleryView: GalleryView;
  imageOrderDir: GalleryOrderDir;
  page: number;
  paginationMode: 'infinite' | 'paginated';
  searchTerm: string;
  starredFirst: boolean;
}

export const getGallerySelectedImageQuery = (values: Record<string, unknown>): GallerySelectedImageQuery => {
  const query =
    values.selectedImageQuery && typeof values.selectedImageQuery === 'object'
      ? (values.selectedImageQuery as Partial<GallerySelectedImageQuery>)
      : null;
  const settings = getGallerySettings(values);

  return {
    boardId:
      query && typeof query.boardId === 'string'
        ? query.boardId
        : typeof values.selectedBoardId === 'string'
          ? values.selectedBoardId
          : 'none',
    galleryView:
      query?.galleryView === 'assets' || query?.galleryView === 'images'
        ? query.galleryView
        : values.galleryView === 'assets'
          ? 'assets'
          : 'images',
    imageOrderDir:
      query?.imageOrderDir === 'ASC' || query?.imageOrderDir === 'DESC' ? query.imageOrderDir : settings.imageOrderDir,
    page:
      query && typeof query.page === 'number' && Number.isFinite(query.page)
        ? Math.max(0, Math.floor(query.page))
        : getGallerySelectedImagePage(values),
    paginationMode:
      query?.paginationMode === 'infinite' || query?.paginationMode === 'paginated'
        ? query.paginationMode
        : settings.paginationMode,
    searchTerm: query && typeof query.searchTerm === 'string' ? query.searchTerm : String(values.searchTerm ?? ''),
    starredFirst: typeof query?.starredFirst === 'boolean' ? query.starredFirst : settings.starredFirst,
  };
};

export const getGalleryTotalImages = (values: Record<string, unknown>): number | null =>
  typeof values.galleryTotalImages === 'number' && Number.isFinite(values.galleryTotalImages)
    ? Math.max(0, values.galleryTotalImages)
    : null;

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

export const getGalleryQueuePlaceholders = (
  queueItems: QueueItem[],
  {
    galleryView,
    imageOrderDir = 'ASC',
    searchTerm,
    selectedBoardId,
  }: { galleryView: GalleryView; imageOrderDir?: GalleryOrderDir; searchTerm: string; selectedBoardId: string }
): GalleryQueuePlaceholder[] => {
  return getVisibleGalleryQueuePlaceholders(getGalleryGenerationSequence(queueItems, null).chronologicalSlots, {
    galleryView,
    imageOrderDir,
    searchTerm,
    selectedBoardId,
  });
};

const getVisibleGalleryQueuePlaceholders = (
  chronologicalSlots: GalleryQueuePlaceholder[],
  {
    galleryView,
    imageOrderDir,
    searchTerm,
    selectedBoardId,
  }: { galleryView: GalleryView; imageOrderDir: GalleryOrderDir; searchTerm: string; selectedBoardId: string }
): GalleryQueuePlaceholder[] => {
  if (galleryView !== 'images' || searchTerm.trim() !== '') {
    return [];
  }

  const placeholders = chronologicalSlots.filter((placeholder) => placeholder.boardId === selectedBoardId);

  return imageOrderDir === 'DESC' ? [...placeholders].reverse() : placeholders;
};

export const getGalleryStateView = (
  values: Record<string, unknown>,
  backendBoards: GalleryBoard[],
  backendImages: GalleryImage[] | null,
  isLoading: boolean,
  queueItems: QueueItem[] = [],
  liveFollowEnabled = false,
  liveTarget: GalleryLiveTarget | null = null
): GalleryStateView => {
  const localImages = getBoundedRecentImages(values.recentImages).map((image) => normalizeGalleryImage(image));
  const images = backendImages ?? (isLoading ? [] : localImages);
  const selectedImageName = typeof values.selectedImageName === 'string' ? values.selectedImageName : null;
  const visibleSelectedImageName = images.some((image) => image.imageName === selectedImageName)
    ? selectedImageName
    : null;
  const selectedImageNames = getGallerySelectedImageNames(values);
  const galleryView = getGalleryView(values);
  const settings = getGallerySettings(values);
  const searchTerm = getGallerySearchTerm(values);
  const boards = backendBoards.length ? backendBoards : [{ ...UNCATEGORIZED_BOARD, imageCount: images.length }];
  const selectedBoardId = getGallerySelectedBoardId(values, backendBoards);
  const compareImage = getGalleryCompareImage(values);
  const compareImageName = compareImage?.imageName ?? null;
  const isComparisonActive =
    visibleSelectedImageName !== null && compareImageName !== null && compareImageName !== visibleSelectedImageName;
  const generationSequence = getGalleryGenerationSequence(queueItems, liveTarget);
  const visibleActivePlaceholder =
    settings.showPendingItems && galleryView === 'images' && searchTerm.trim() === ''
      ? generationSequence.liveSlot?.boardId === selectedBoardId
        ? generationSequence.liveSlot
        : null
      : null;
  const currentItem = getGalleryCurrentItem({
    activePlaceholder: visibleActivePlaceholder,
    isComparisonActive,
    liveFollowEnabled,
    selectedImageName: visibleSelectedImageName,
  });

  return {
    boards,
    compareImageName,
    currentItem,
    galleryView,
    images,
    isLoading,
    pendingPlaceholders: settings.showPendingItems
      ? getVisibleGalleryQueuePlaceholders(generationSequence.chronologicalSlots, {
          galleryView,
          imageOrderDir: settings.imageOrderDir,
          searchTerm,
          selectedBoardId,
        })
      : [],
    projectBoardId: getGalleryProjectBoardId(values),
    searchTerm,
    selectedBoardId,
    selectedImageName: visibleSelectedImageName,
    selectedImageNames:
      visibleSelectedImageName && !selectedImageNames.includes(visibleSelectedImageName)
        ? [visibleSelectedImageName, ...selectedImageNames]
        : selectedImageNames,
    settings,
  };
};

export const getBoardCounts = (board: GalleryBoard): { assetCount: number; imageCount: number } => ({
  assetCount: board.assetCount,
  imageCount: board.imageCount,
});

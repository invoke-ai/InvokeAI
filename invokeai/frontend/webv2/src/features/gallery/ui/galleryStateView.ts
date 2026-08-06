import type { GalleryBoard, GalleryImage, GalleryOrderDir, GalleryView } from '@features/gallery/core/types';
import type { QueueItem } from '@features/queue/contracts';

import {
  legacyGeneratedImageToGalleryItem,
  toGalleryItemKey,
  type GalleryItem,
  type GalleryItemKey,
} from '@features/gallery/core/items';
import { getBoundedRecentImages } from '@features/gallery/core/recentImages';
import {
  getPersistedSelectedGalleryItemKeys,
  getSelectedGalleryImageFromValues,
  getSelectedGalleryItemFromValues,
} from '@features/gallery/core/selection';
import { getGallerySettings, type GallerySettings } from '@features/gallery/core/settings';
import { getQueueItemSnapshotBatchCount, getQueueItemSnapshotDimensions } from '@features/queue/contracts';

/**
 * Stand-in shown before any board has loaded. `name` is intentionally empty:
 * the UI labels uncategorized boards from `kind` through `getGalleryBoardLabel`,
 * so there is no English string to invent here.
 */
const UNCATEGORIZED_BOARD: GalleryBoard = {
  archived: false,
  assetCount: 0,
  id: 'none',
  imageCount: 0,
  kind: 'uncategorized',
  name: '',
  projectId: null,
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
  | { kind: 'item'; itemKey: GalleryItemKey }
  | { kind: 'placeholder'; placeholder: GalleryQueuePlaceholder }
  | null;

export interface GalleryStateView {
  boards: GalleryBoard[];
  compareImageKey: GalleryItemKey | null;
  currentItem: GalleryCurrentItem;
  galleryView: GalleryView;
  items: GalleryItem[];
  isLoading: boolean;
  pendingPlaceholders: GalleryQueuePlaceholder[];
  projectBoardId: string | null;
  searchTerm: string;
  selectedBoardId: string;
  selectedItemKey: GalleryItemKey | null;
  selectedItemKeys: GalleryItemKey[];
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
  selectedItemKey,
}: {
  activePlaceholder: GalleryQueuePlaceholder | null;
  isComparisonActive: boolean;
  liveFollowEnabled: boolean;
  selectedItemKey: GalleryItemKey | null;
}): GalleryCurrentItem => {
  if (liveFollowEnabled && !isComparisonActive && activePlaceholder) {
    return { kind: 'placeholder', placeholder: activePlaceholder };
  }

  return selectedItemKey ? { itemKey: selectedItemKey, kind: 'item' } : null;
};

export const getGalleryView = (values: Record<string, unknown>): GalleryView =>
  values.galleryView === 'assets' ? 'assets' : 'images';

export const getGallerySearchTerm = (values: Record<string, unknown>): string =>
  typeof values.searchTerm === 'string' ? values.searchTerm : '';

/**
 * Where new results land, resolved against the boards this install actually has.
 *
 * A saved selection survives whenever it still resolves — it is a deliberate choice, and a project
 * opened where its destination still exists should keep using it. When it does not resolve the
 * project's own board is the better answer than Uncategorized: a project arriving from another
 * install, or one whose pre-migration board was rejected as ambiguous, names a board id that means
 * nothing here, and dropping it to Uncategorized would quietly scatter that project's output.
 *
 * An empty board list means "still loading", not "no such board", so nothing is resolved yet.
 */
export const getGallerySelectedBoardId = (values: Record<string, unknown>, backendBoards: GalleryBoard[]): string => {
  const selectedBoardId = typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none';

  if (backendBoards.length === 0 || backendBoards.some((board) => board.id === selectedBoardId)) {
    return selectedBoardId;
  }

  const projectBoardId = getGalleryProjectBoardId(values);

  if (projectBoardId !== null && backendBoards.some((board) => board.id === projectBoardId)) {
    return projectBoardId;
  }

  return 'none';
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

export const getGalleryCompareImage = (values: Record<string, unknown>): GalleryImage | null =>
  getSelectedGalleryImageFromValues({
    selectedBoardId: values.selectedBoardId,
    selectedImage: values.compareImage,
    selectedImageName: null,
  });

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
  backendItems: GalleryItem[] | null,
  isLoading: boolean,
  queueItems: QueueItem[] = [],
  liveFollowEnabled = false,
  liveTarget: GalleryLiveTarget | null = null
): GalleryStateView => {
  const localItems = getBoundedRecentImages(values.recentImages).map(legacyGeneratedImageToGalleryItem);
  const items = backendItems ?? (isLoading ? [] : localItems);
  const selectedItem = getSelectedGalleryItemFromValues(values);
  const persistedSelectedItemKey =
    typeof values.selectedImageName === 'string'
      ? (getPersistedSelectedGalleryItemKeys({ selectedImageName: values.selectedImageName })[0] ?? null)
      : selectedItem
        ? toGalleryItemKey(selectedItem)
        : null;
  const visibleSelectedItemKey =
    persistedSelectedItemKey && items.some((item) => toGalleryItemKey(item) === persistedSelectedItemKey)
      ? persistedSelectedItemKey
      : null;
  const selectedItemKeys = getPersistedSelectedGalleryItemKeys(values);
  const galleryView = getGalleryView(values);
  const settings = getGallerySettings(values);
  const searchTerm = getGallerySearchTerm(values);
  const boards = backendBoards.length
    ? backendBoards
    : [
        {
          ...UNCATEGORIZED_BOARD,
          imageCount: items.filter((item) => item.kind === 'image' && item.category === 'general').length,
          projectId: null,
          videoCount: items.filter((item) => item.kind === 'video').length,
        },
      ];
  const selectedBoardId = getGallerySelectedBoardId(values, backendBoards);
  const compareImage = getGalleryCompareImage(values);
  const compareImageKey = compareImage ? toGalleryItemKey({ kind: 'image', name: compareImage.imageName }) : null;
  const isComparisonActive =
    visibleSelectedItemKey?.startsWith('image:') === true &&
    compareImageKey !== null &&
    compareImageKey !== visibleSelectedItemKey;
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
    selectedItemKey: visibleSelectedItemKey,
  });

  return {
    boards,
    compareImageKey,
    currentItem,
    galleryView,
    items,
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
    selectedItemKey: visibleSelectedItemKey,
    selectedItemKeys:
      visibleSelectedItemKey && !selectedItemKeys.includes(visibleSelectedItemKey)
        ? [visibleSelectedItemKey, ...selectedItemKeys]
        : selectedItemKeys,
    settings,
  };
};

export const getBoardCounts = (
  board: GalleryBoard
): { assetCount: number; imageCount: number; videoCount: number } => ({
  assetCount: board.assetCount,
  imageCount: board.imageCount,
  videoCount: board.videoCount,
});

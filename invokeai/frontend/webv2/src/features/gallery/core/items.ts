import type { GalleryImage, GalleryOrderDir, GeneratedImageContract } from './types';

export type GalleryItemKind = 'image' | 'video';

export type GalleryItemCategory = GalleryImage['imageCategory'];

export interface GalleryItemRef {
  kind: GalleryItemKind;
  name: string;
}

export type GalleryItemKey = `${GalleryItemKind}:${string}`;

interface GalleryItemBase {
  boardId: string;
  category: GalleryItemCategory;
  createdAt: string;
  fullUrl: string;
  height: number;
  isIntermediate: boolean;
  name: string;
  starred: boolean;
  thumbnailUrl: string;
  width: number;
}

export interface GalleryImageItem extends GalleryItemBase {
  kind: 'image';
  sourceQueueItemId?: string;
}

export interface GalleryVideoItem extends GalleryItemBase {
  durationSeconds: number;
  fps?: number;
  kind: 'video';
}

export type GalleryItem = GalleryImageItem | GalleryVideoItem;

export interface GalleryItemsPage {
  items: GalleryItem[];
  total: number;
}

export interface GalleryItemMutationResult {
  /** Boards whose contents/counts changed for at least one confirmed success. */
  affectedBoardIds?: string[];
  failed: GalleryItemRef[];
  succeeded: GalleryItemRef[];
}

export const toGalleryItemKey = ({ kind, name }: GalleryItemRef): GalleryItemKey => `${kind}:${name}`;

export const shouldStarSelection = (items: readonly GalleryItem[], refs: readonly GalleryItemRef[]): boolean => {
  if (refs.length === 0) {
    return false;
  }

  const loadedItemsByKey = new Map(items.map((item) => [toGalleryItemKey(item), item]));
  return refs.some((ref) => !loadedItemsByKey.get(toGalleryItemKey(ref))?.starred);
};

export const parseGalleryItemKey = (key: string): GalleryItemRef => {
  const separatorIndex = key.indexOf(':');
  const kind = key.slice(0, separatorIndex);
  const name = key.slice(separatorIndex + 1);

  if ((kind === 'image' || kind === 'video') && name.length > 0) {
    return { kind, name };
  }

  return { kind: 'image', name: key };
};

export const toGalleryItemRef = ({ kind, name }: GalleryItem): GalleryItemRef => ({ kind, name });

export const isGalleryImageItem = (item: GalleryItem): item is GalleryImageItem => item.kind === 'image';

export const assertNeverGalleryItem = (item: never): never => {
  throw new Error(`Unexpected gallery item: ${String(item)}`);
};

const compareSqliteBinaryText = (a: string, b: string): number => (a === b ? 0 : a < b ? -1 : 1);

/** Mirrors the backend's starred/time/kind/name order for mixed gallery items. */
export const compareGalleryItems = (
  a: GalleryItem,
  b: GalleryItem,
  { orderDir = 'DESC', starredFirst = false }: { orderDir?: GalleryOrderDir; starredFirst?: boolean }
): number => {
  if (starredFirst && a.starred !== b.starred) {
    return a.starred ? -1 : 1;
  }

  const direction = orderDir === 'ASC' ? 1 : -1;
  const chronologicalOrder = compareSqliteBinaryText(a.createdAt, b.createdAt);

  if (chronologicalOrder !== 0) {
    return direction * chronologicalOrder;
  }

  const kindOrder = compareSqliteBinaryText(a.kind, b.kind);

  if (kindOrder !== 0) {
    return direction * kindOrder;
  }

  return direction * compareSqliteBinaryText(a.name, b.name);
};

type LegacyGalleryImage = GeneratedImageContract & Partial<Pick<GalleryImage, 'boardId' | 'imageCategory' | 'starred'>>;

export const legacyGeneratedImageToGalleryItem = (image: LegacyGalleryImage): GalleryImageItem => ({
  boardId: image.boardId ?? 'none',
  category: image.imageCategory ?? 'general',
  createdAt: image.queuedAt,
  fullUrl: image.imageUrl,
  height: image.height,
  isIntermediate: false,
  kind: 'image',
  name: image.imageName,
  sourceQueueItemId: image.sourceQueueItemId,
  starred: image.starred ?? false,
  thumbnailUrl: image.thumbnailUrl,
  width: image.width,
});

export const galleryImageItemToGalleryImage = (item: GalleryImageItem): GalleryImage => ({
  boardId: item.boardId,
  height: item.height,
  imageCategory: item.category,
  imageName: item.name,
  imageUrl: item.fullUrl,
  queuedAt: item.createdAt,
  sourceQueueItemId: item.sourceQueueItemId ?? 'backend-gallery',
  starred: item.starred,
  thumbnailUrl: item.thumbnailUrl,
  width: item.width,
});

export const formatGalleryVideoDuration = (durationSeconds: number): string => {
  if (!Number.isFinite(durationSeconds) || durationSeconds < 0) {
    return '0:00';
  }

  const totalSeconds = Math.ceil(durationSeconds);
  const seconds = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);

  if (totalMinutes < 60) {
    return `${totalMinutes}:${String(seconds).padStart(2, '0')}`;
  }

  const minutes = totalMinutes % 60;
  const hours = Math.floor(totalMinutes / 60);
  return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
};

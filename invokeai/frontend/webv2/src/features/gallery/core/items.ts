import { normalizeServerTimestamp } from '@platform/time/serverTimestamp';

import type {
  GalleryImage,
  GalleryOrderDir,
  GalleryView,
  GeneratedImageContract,
  GeneratedVideoContract,
} from './types';

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
  /** Video records say; the mixed-media listing does not. */
  hasWorkflow?: boolean;
  kind: 'video';
  /** `audio_upload` marks audio converted into a waveform video. */
  mediaOrigin?: string;
}

export type GalleryItem = GalleryImageItem | GalleryVideoItem;

export interface GalleryItemsPage {
  items: GalleryItem[];
  /** Omitted only by legacy fixtures/cache entries. */
  offset?: number;
  /** Aligned with `items`; sparse hydration leaves index gaps. */
  itemIndices?: number[];
  total: number;
}

export interface GalleryItemMutationResult {
  affectedBoardIds?: string[];
  failed: GalleryItemRef[];
  succeeded: GalleryItemRef[];
}

export const toGalleryItemKey = ({ kind, name }: GalleryItemRef): GalleryItemKey => `${kind}:${name}`;

/** Mirrors the backend's `IMAGE_CATEGORIES`/`ASSETS_CATEGORIES`; `other` marks Canvas-owned pixels no view lists. */
const galleryViewByCategory: Record<GalleryItemCategory, GalleryView | null> = {
  control: 'assets',
  general: 'images',
  mask: 'assets',
  other: null,
  user: 'assets',
};

/** Null for a Canvas-owned item: callers must not open a view for it, as the listing would never contain it. */
export const getGalleryItemView = ({ category }: Pick<GalleryItem, 'category'>): GalleryView | null =>
  galleryViewByCategory[category];

export const getGalleryViewCategories = (view: GalleryView): GalleryItemCategory[] =>
  (Object.keys(galleryViewByCategory) as GalleryItemCategory[]).filter(
    (category) => galleryViewByCategory[category] === view
  );

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

/** SQLite and ISO separators would otherwise misorder same-day items. */
export const compareCreatedAt = (a: string, b: string): number =>
  compareSqliteBinaryText(normalizeServerTimestamp(a), normalizeServerTimestamp(b));

/** Mirrors the backend's time/kind/name order. */
export const compareGalleryItems = (
  a: GalleryItem,
  b: GalleryItem,
  { orderDir = 'DESC' }: { orderDir?: GalleryOrderDir } = {}
): number => {
  const direction = orderDir === 'ASC' ? 1 : -1;
  const chronologicalOrder = compareCreatedAt(a.createdAt, b.createdAt);

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
  createdAt: image.createdAt ?? image.queuedAt,
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

export const generatedVideoToGalleryItem = (video: GeneratedVideoContract): GalleryVideoItem => ({
  boardId: video.boardId ?? 'none',
  category: video.category,
  createdAt: video.createdAt ?? video.queuedAt,
  durationSeconds: video.durationSeconds,
  ...(video.fps === undefined ? {} : { fps: video.fps }),
  fullUrl: video.videoUrl,
  height: video.height,
  isIntermediate: video.isIntermediate,
  kind: 'video',
  ...(video.mediaOrigin === undefined ? {} : { mediaOrigin: video.mediaOrigin }),
  name: video.videoName,
  starred: false,
  thumbnailUrl: video.thumbnailUrl,
  width: video.width,
});

export const galleryImageItemToGalleryImage = (item: GalleryImageItem): GalleryImage => ({
  boardId: item.boardId,
  createdAt: item.createdAt,
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

/** MIME wildcards mirror server checks; images offer only round-trippable formats. */
const GALLERY_UPLOAD_FORMATS: Record<GalleryItemKind, { extensions: readonly string[]; mimes: readonly string[] }> = {
  image: {
    extensions: ['.png', '.jpg', '.jpeg', '.webp'],
    mimes: ['image/png', 'image/jpeg', 'image/webp'],
  },
  video: {
    extensions: [
      '.mp4',
      '.mov',
      '.m4v',
      '.webm',
      '.mkv',
      '.avi',
      '.mpg',
      '.mpeg',
      '.3gp',
      '.wmv',
      '.asf',
      '.mp3',
      '.m4a',
      '.aac',
      '.wav',
      '.flac',
      '.ogg',
      '.oga',
      '.opus',
      '.aiff',
      '.aif',
      '.wma',
    ],
    mimes: ['video/*', 'audio/*'],
  },
};

/** Keeps top-level declarations tree-shakeable, so this module's barrels stay out. */
const GALLERY_UPLOAD_KINDS = ['image', 'video'] as const;

/** Advisory only (browsers offer "All files"), so callers classify. */
export const getGalleryUploadAccept = (kinds: readonly GalleryItemKind[]): string =>
  kinds
    .flatMap((kind) => [...GALLERY_UPLOAD_FORMATS[kind].mimes, ...GALLERY_UPLOAD_FORMATS[kind].extensions])
    .join(',');

/** Exact MIME beats wildcard beats filename, so a file the OS typed is never routed by its extension. */
export const classifyGalleryUpload = (file: Pick<File, 'name' | 'type'>): { kind: GalleryItemKind } | null => {
  const mimeType = file.type.toLowerCase();

  // The legacy alias some Windows tools emit: accepted, never advertised.
  if (mimeType === 'image/jpg') {
    return { kind: 'image' };
  }
  for (const kind of GALLERY_UPLOAD_KINDS) {
    if (GALLERY_UPLOAD_FORMATS[kind].mimes.includes(mimeType)) {
      return { kind };
    }
  }
  for (const kind of GALLERY_UPLOAD_KINDS) {
    // Mirrors ACCEPTED_*_MIME_PREFIXES on the routes.
    if (
      GALLERY_UPLOAD_FORMATS[kind].mimes.some((mime) => mime.endsWith('/*') && mimeType.startsWith(mime.slice(0, -1)))
    ) {
      return { kind };
    }
  }

  const lowerName = file.name.toLowerCase();

  for (const kind of GALLERY_UPLOAD_KINDS) {
    if (GALLERY_UPLOAD_FORMATS[kind].extensions.some((extension) => lowerName.endsWith(extension))) {
      return { kind };
    }
  }

  return null;
};

/** They list items but can never receive them. */
export const DATE_BOARD_ID_PREFIX = 'by_date:';

/** A request scope, never stored on. */
export const ALL_READABLE_BOARDS_ID = 'all';

export const isDateBoardId = (boardId: string): boolean => boardId.startsWith(DATE_BOARD_ID_PREFIX);

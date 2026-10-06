import { normalizeServerTimestamp } from '@platform/time/serverTimestamp';

import type { GalleryImage, GalleryOrderDir, GeneratedImageContract, GeneratedVideoContract } from './types';

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
  /** `audio_upload` identifies audio converted into a waveform video rather than footage. */
  mediaOrigin?: string;
}

export type GalleryItem = GalleryImageItem | GalleryVideoItem;

export interface GalleryItemsPage {
  items: GalleryItem[];
  total: number;
}

export interface GalleryItemMutationResult {
  /** Boards whose contents/counts changed for at least one confirmed success. */
  affectedBoardIds?: string[];
  /** Every requested item that was not changed, `starredSkipped` included. */
  failed: GalleryItemRef[];
  /** Starred items a delete left alone because protection was on. They still exist; this is not a failure. */
  starredSkipped?: GalleryItemRef[];
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

/**
 * Normalize SQLite and ISO timestamps before comparing; their space/T separators otherwise misorder same-day
 * items.
 */
const compareCreatedAt = (a: string, b: string): number =>
  compareSqliteBinaryText(normalizeServerTimestamp(a), normalizeServerTimestamp(b));

/** Mirrors the backend's time/kind/name order for mixed gallery items. */
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

/** Share upload acceptance and classification here to prevent drift without adding an initial bundle request. */

/**
 * Audio uploads become waveform videos. Video/audio MIME wildcards mirror server checks; extensions cover unknown
 * MIME types. Images deliberately offer only round-trippable formats.
 */
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

/**
 * Keep top-level declarations tree-shakeable; scanning the small format table avoids retaining this module's
 * barrels.
 */
const GALLERY_UPLOAD_KINDS = ['image', 'video'] as const;

/**
 * The file input `accept` list for the given kinds. Advisory only — every browser offers an
 * "All files" escape hatch, so callers still classify what comes back.
 */
export const getGalleryUploadAccept = (kinds: readonly GalleryItemKind[]): string =>
  kinds
    .flatMap((kind) => [...GALLERY_UPLOAD_FORMATS[kind].mimes, ...GALLERY_UPLOAD_FORMATS[kind].extensions])
    .join(',');

/**
 * Which upload route a picked file belongs to, or null when no route takes it. An exact MIME
 * match wins over a wildcard, and both win over the filename, so a file the OS typed is never
 * routed by its extension.
 */
export const classifyGalleryUpload = (file: Pick<File, 'name' | 'type'>): { kind: GalleryItemKind } | null => {
  const mimeType = file.type.toLowerCase();

  // The legacy alias some Windows tools emit; the image route accepts it, but no picker
  // needs to advertise it, so it is classified without being offered.
  if (mimeType === 'image/jpg') {
    return { kind: 'image' };
  }
  for (const kind of GALLERY_UPLOAD_KINDS) {
    if (GALLERY_UPLOAD_FORMATS[kind].mimes.includes(mimeType)) {
      return { kind };
    }
  }
  for (const kind of GALLERY_UPLOAD_KINDS) {
    // `video/*` matches any `video/` type, mirroring ACCEPTED_*_MIME_PREFIXES on the routes.
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

/** Virtual "by date" boards (`by_date:<YYYY-MM-DD>`) list items but can never receive them. */
export const DATE_BOARD_ID_PREFIX = 'by_date:';

export const isDateBoardId = (boardId: string): boolean => boardId.startsWith(DATE_BOARD_ID_PREFIX);

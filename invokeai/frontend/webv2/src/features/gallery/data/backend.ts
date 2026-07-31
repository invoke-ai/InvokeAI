import type {
  GalleryImageItem,
  GalleryItem,
  GalleryItemCategory,
  GalleryItemRef,
  GalleryItemsPage,
  GalleryVideoItem,
} from '@features/gallery/core/items';
import type {
  GalleryBoard,
  GalleryBoardDeletionResult,
  GalleryBoardOrderBy,
  GalleryDeletionResult,
  GalleryImage,
  GalleryImageMetadata,
  GalleryImagesPage,
  GalleryOrderDir,
  GalleryView,
} from '@features/gallery/core/types';

import { isTimestampInRange } from '@platform/search/dateTokens';
import {
  AccountScopeExpiredError,
  assertAccountScopeCurrent,
  captureAccountScope,
} from '@platform/state/accountLifecycle';
import {
  absolutizeApiUrl,
  ApiError,
  apiFetchJson,
  apiFetchRaw,
  HttpRequestIdentityExpiredError,
  sleep,
} from '@platform/transport/http';

import { getGalleryImageThumbnailUrl } from './imageUrls';
import { getGalleryVideoThumbnailUrl } from './videoUrls';

interface BackendBoardDTO {
  board_id: string;
  board_name: string;
  image_count: number;
  asset_count: number;
  /** Videos on the board. Counted separately from `image_count` by the backend. */
  video_count?: number;
  archived: boolean;
  cover_image_name?: string | null;
  /** Set instead of `cover_image_name` when the board's most recent item is a video. */
  cover_video_name?: string | null;
  created_at?: string | null;
  /** Board owner's display name; populated only for admins on multi-user backends. */
  owner_username?: string | null;
}

/**
 * 'board' is a real backend board; 'uncategorized' is the pseudo-board for
 * unassigned images (board_id 'none'); 'date' is a read-only virtual board
 * grouping images by creation date (id 'by_date:YYYY-MM-DD').
 */
const DATE_BOARD_ID_PREFIX = 'by_date:';
export const ALL_READABLE_BOARDS_ID = 'all';

export const isDateBoardId = (boardId: string): boolean => boardId.startsWith(DATE_BOARD_ID_PREFIX);

const getDateFromBoardId = (boardId: string): string => boardId.slice(DATE_BOARD_ID_PREFIX.length);

const getUploadBoardId = (boardId: string): string | undefined =>
  boardId === 'none' || boardId === ALL_READABLE_BOARDS_ID || isDateBoardId(boardId) ? undefined : boardId;

export type GalleryUploadKind = 'image' | 'video';

const GALLERY_UPLOAD_KIND_BY_MIME = new Map<string, GalleryUploadKind>([
  ['image/jpeg', 'image'],
  ['image/jpg', 'image'],
  ['image/png', 'image'],
  ['image/webp', 'image'],
  ['video/mp4', 'video'],
]);

const GALLERY_UPLOAD_KIND_BY_EXTENSION = new Map<string, GalleryUploadKind>([
  ['.jpeg', 'image'],
  ['.jpg', 'image'],
  ['.png', 'image'],
  ['.webp', 'image'],
  ['.mp4', 'video'],
]);

export const classifyGalleryUpload = (file: Pick<File, 'name' | 'type'>): { kind: GalleryUploadKind } | null => {
  const mimeKind = GALLERY_UPLOAD_KIND_BY_MIME.get(file.type.toLowerCase());

  if (mimeKind) {
    return { kind: mimeKind };
  }

  const lowerName = file.name.toLowerCase();

  for (const [extension, kind] of GALLERY_UPLOAD_KIND_BY_EXTENSION) {
    if (lowerName.endsWith(extension)) {
      return { kind };
    }
  }

  return null;
};

interface BackendImageDTO {
  image_name: string;
  image_url: string;
  thumbnail_url: string;
  width: number;
  height: number;
  created_at: string;
  image_category: 'general' | 'control' | 'mask' | 'user' | 'other';
  is_intermediate: boolean;
  starred?: boolean;
  board_id?: string | null;
}

export interface BackendGalleryItemDTO {
  board_id?: string | null;
  category: GalleryItemCategory;
  created_at: string;
  duration?: number | null;
  fps?: number | null;
  full_url: string;
  height: number;
  is_intermediate: boolean;
  kind: 'image' | 'video';
  name: string;
  starred: boolean;
  thumbnail_url: string;
  width: number;
}

interface BackendVideoDTO {
  board_id?: string | null;
  created_at: string;
  duration: number;
  fps?: number | null;
  height: number;
  is_intermediate: boolean;
  starred: boolean;
  thumbnail_url: string;
  video_category: GalleryItemCategory;
  video_name: string;
  video_url: string;
  width: number;
}

interface ListImagesResponse {
  items: BackendImageDTO[];
  limit: number;
  offset: number;
  total: number;
}

/**
 * Mirrors the backend's `IMAGE_CATEGORIES` / `ASSETS_CATEGORIES`
 * (`image_records_common.py`). `'other'` is in neither: it is the private
 * category for images a canvas layer owns, which are layer pixels rather than
 * gallery content and so belong to neither view.
 */
const imageCategories = ['general'];
const assetCategories = ['control', 'mask', 'user'];

const toSearchParams = (entries: Record<string, boolean | number | string | string[] | undefined>): string => {
  const params = new URLSearchParams();

  for (const [key, value] of Object.entries(entries)) {
    if (value === undefined || value === '') {
      continue;
    }

    if (Array.isArray(value)) {
      for (const item of value) {
        params.append(key, item);
      }
      continue;
    }

    params.set(key, String(value));
  }

  return params.toString();
};

/**
 * The backend sets exactly one of `cover_image_name` / `cover_video_name`, depending on which
 * kind the board's most recent item is. Both resolve to a static WebP thumbnail, so the cover
 * renders identically either way.
 */
const getBoardCoverThumbnailUrl = (
  board: Pick<BackendBoardDTO, 'cover_image_name' | 'cover_video_name'>
): string | undefined => {
  if (board.cover_image_name) {
    return getGalleryImageThumbnailUrl(board.cover_image_name);
  }

  return board.cover_video_name ? getGalleryVideoThumbnailUrl(board.cover_video_name) : undefined;
};

const mapBoard = (board: BackendBoardDTO): GalleryBoard => ({
  archived: board.archived,
  assetCount: board.asset_count,
  coverImageName: board.cover_image_name,
  coverThumbnailUrl: getBoardCoverThumbnailUrl(board),
  coverVideoName: board.cover_video_name,
  createdAt: board.created_at ?? null,
  id: board.board_id,
  imageCount: board.image_count,
  kind: 'board',
  name: board.board_name,
  ownerName: board.owner_username ?? null,
  videoCount: board.video_count ?? 0,
});

const getGalleryTotal = async ({
  boardId,
  categories,
  signal,
}: {
  boardId: string;
  categories: string[];
  signal?: AbortSignal;
}): Promise<number> => {
  const query = toSearchParams({
    board_id: boardId,
    categories,
    is_intermediate: false,
    limit: 0,
    offset: 0,
  });
  const body = await apiFetchJson<Pick<ListImagesResponse, 'total'>>(`/api/v1/images/?${query}`, { signal });

  return body.total;
};

/**
 * Videos carry no category split — the gallery's Images/Assets views are an image-only
 * distinction — so this counts every non-intermediate video on the board.
 */
const getGalleryVideoTotal = async ({
  boardId,
  signal,
}: {
  boardId: string;
  signal?: AbortSignal;
}): Promise<number> => {
  const query = toSearchParams({ board_id: boardId, is_intermediate: false, limit: 0, offset: 0 });
  const body = await apiFetchJson<{ total: number }>(`/api/v1/videos/?${query}`, { signal });

  return body.total;
};

const mapImage = (image: BackendImageDTO): GalleryImage => ({
  boardId: image.board_id ?? 'none',
  height: image.height,
  imageCategory: image.image_category,
  imageName: image.image_name,
  imageUrl: absolutizeApiUrl(image.image_url),
  queuedAt: image.created_at,
  sourceQueueItemId: 'backend-gallery',
  starred: image.starred ?? false,
  thumbnailUrl: absolutizeApiUrl(image.thumbnail_url),
  width: image.width,
});

const mapGalleryItemBase = (
  item: BackendGalleryItemDTO
): Omit<GalleryItem, 'durationSeconds' | 'fps' | 'kind' | 'sourceQueueItemId'> => ({
  boardId: item.board_id ?? 'none',
  category: item.category,
  createdAt: item.created_at,
  fullUrl: absolutizeApiUrl(item.full_url),
  height: item.height,
  isIntermediate: item.is_intermediate,
  name: item.name,
  starred: item.starred,
  thumbnailUrl: absolutizeApiUrl(item.thumbnail_url),
  width: item.width,
});

const mapGalleryItem = (item: BackendGalleryItemDTO): GalleryItem => {
  const base = mapGalleryItemBase(item);

  if (item.kind === 'image') {
    return { ...base, kind: 'image' };
  }

  if (typeof item.duration !== 'number' || !Number.isFinite(item.duration)) {
    throw new TypeError(`Gallery video "${item.name}" must have a finite duration.`);
  }

  return {
    ...base,
    durationSeconds: item.duration,
    ...(item.fps === null || item.fps === undefined ? {} : { fps: item.fps }),
    kind: 'video',
  };
};

const mapBackendImageToGalleryItem = (image: BackendImageDTO): GalleryImageItem => ({
  boardId: image.board_id ?? 'none',
  category: image.image_category,
  createdAt: image.created_at,
  fullUrl: absolutizeApiUrl(image.image_url),
  height: image.height,
  isIntermediate: image.is_intermediate,
  kind: 'image',
  name: image.image_name,
  sourceQueueItemId: 'backend-gallery',
  starred: image.starred ?? false,
  thumbnailUrl: absolutizeApiUrl(image.thumbnail_url),
  width: image.width,
});

const mapVideo = (video: BackendVideoDTO): GalleryVideoItem => {
  if (!Number.isFinite(video.duration)) {
    throw new TypeError(`Gallery video "${video.video_name}" must have a finite duration.`);
  }

  return {
    boardId: video.board_id ?? 'none',
    category: video.video_category,
    createdAt: video.created_at,
    durationSeconds: video.duration,
    ...(video.fps === null || video.fps === undefined ? {} : { fps: video.fps }),
    fullUrl: absolutizeApiUrl(video.video_url),
    height: video.height,
    isIntermediate: video.is_intermediate,
    kind: 'video',
    name: video.video_name,
    starred: video.starred,
    thumbnailUrl: absolutizeApiUrl(video.thumbnail_url),
    width: video.width,
  };
};

const normalizeTotal = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.max(0, value) : Math.max(0, fallback);

export const listGalleryBoards = async ({
  includeArchived = false,
  orderBy = 'created_at',
  orderDir = 'DESC',
  signal,
}: {
  includeArchived?: boolean;
  orderBy?: GalleryBoardOrderBy;
  orderDir?: GalleryOrderDir;
  signal?: AbortSignal;
} = {}): Promise<GalleryBoard[]> => {
  const boardsQuery = toSearchParams({
    all: true,
    direction: orderDir,
    include_archived: includeArchived,
    order_by: orderBy,
  });
  const boardsBodyPromise = apiFetchJson<BackendBoardDTO[] | { items?: BackendBoardDTO[] }>(
    `/api/v1/boards/?${boardsQuery}`,
    { signal }
  );

  const [body, uncategorizedImageCount, uncategorizedAssetCount, uncategorizedVideoCount] = await Promise.all([
    boardsBodyPromise,
    getGalleryTotal({ boardId: 'none', categories: imageCategories, signal }),
    getGalleryTotal({ boardId: 'none', categories: assetCategories, signal }),
    // Real boards carry `video_count` in their DTO, but the uncategorized pseudo-board is
    // assembled here, so its video total needs its own request.
    getGalleryVideoTotal({ boardId: 'none', signal }),
  ]);
  const boards = Array.isArray(body) ? body : (body.items ?? []);

  return [
    {
      archived: false,
      assetCount: uncategorizedAssetCount,
      id: 'none',
      imageCount: uncategorizedImageCount,
      kind: 'uncategorized',
      name: 'Uncategorized',
      videoCount: uncategorizedVideoCount,
    },
    ...boards.filter((board) => includeArchived || !board.archived).map(mapBoard),
  ];
};

/**
 * Date boards are now derived from the polymorphic gallery service, so a date can exist
 * because of videos alone — its `image_count` may be 0 while the board is non-empty.
 */
interface VirtualDateBoardDTO {
  virtual_board_id: string;
  board_name: string;
  date: string;
  image_count: number;
  asset_count: number;
  video_count?: number;
  cover_image_name?: string | null;
  cover_video_name?: string | null;
}

export const listGalleryDateBoards = async (signal?: AbortSignal): Promise<GalleryBoard[]> => {
  const body = await apiFetchJson<VirtualDateBoardDTO[]>('/api/v1/virtual_boards/by_date', { signal });

  return body.map((board) => ({
    archived: false,
    assetCount: board.asset_count,
    coverImageName: board.cover_image_name,
    coverThumbnailUrl: getBoardCoverThumbnailUrl(board),
    coverVideoName: board.cover_video_name,
    id: board.virtual_board_id,
    imageCount: board.image_count,
    kind: 'date',
    name: board.board_name,
    videoCount: board.video_count ?? 0,
  }));
};

export const getGalleryImagesByNames = async (imageNames: string[], signal?: AbortSignal): Promise<GalleryImage[]> => {
  if (imageNames.length === 0) {
    return [];
  }

  const body = await apiFetchJson<BackendImageDTO[]>('/api/v1/images/images_by_names', {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });

  const imagesByName = new Map(body.map((image) => [image.image_name, mapImage(image)]));

  return imageNames.flatMap((imageName) => {
    const image = imagesByName.get(imageName);

    return image ? [image] : [];
  });
};

export const getGalleryImageItemsByNames = async (
  imageNames: string[],
  signal?: AbortSignal
): Promise<GalleryImageItem[]> => {
  if (imageNames.length === 0) {
    return [];
  }

  const body = await apiFetchJson<BackendImageDTO[]>('/api/v1/images/images_by_names', {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });
  const imagesByName = new Map(body.map((image) => [image.image_name, mapBackendImageToGalleryItem(image)]));

  return imageNames.flatMap((imageName) => {
    const image = imagesByName.get(imageName);

    return image ? [image] : [];
  });
};

export const getGalleryImageByName = async (imageName: string, signal?: AbortSignal): Promise<GalleryImage> => {
  const body = await apiFetchJson<BackendImageDTO>(`/api/v1/images/i/${encodeURIComponent(imageName)}`, { signal });

  return mapImage(body);
};

export const getGalleryVideoByName = async (videoName: string, signal?: AbortSignal): Promise<GalleryVideoItem> => {
  const body = await apiFetchJson<BackendVideoDTO>(`/api/v1/videos/i/${encodeURIComponent(videoName)}`, { signal });

  return mapVideo(body);
};

export const getGalleryItemByRef = async (ref: GalleryItemRef, signal?: AbortSignal): Promise<GalleryItem> => {
  if (ref.kind === 'image') {
    const body = await apiFetchJson<BackendImageDTO>(`/api/v1/images/i/${encodeURIComponent(ref.name)}`, { signal });

    return mapBackendImageToGalleryItem(body);
  }

  return getGalleryVideoByName(ref.name, signal);
};

export const getGalleryVideoMetadata = async (
  videoName: string,
  signal?: AbortSignal
): Promise<Record<string, unknown> | null> => {
  const body = await apiFetchJson<unknown>(`/api/v1/videos/i/${encodeURIComponent(videoName)}/metadata`, { signal });

  return body && typeof body === 'object' && !Array.isArray(body) ? (body as Record<string, unknown>) : null;
};

export interface GalleryVideoWorkflow {
  graph: string | null;
  workflow: string | null;
}

export const getGalleryVideoWorkflow = (videoName: string, signal?: AbortSignal): Promise<GalleryVideoWorkflow> =>
  apiFetchJson<GalleryVideoWorkflow>(`/api/v1/videos/i/${encodeURIComponent(videoName)}/workflow`, { signal });

interface PaletteDateBoardImageNames {
  imageNames: string[];
  total: number;
}

/**
 * Date virtual boards have no offset-paginated DTO endpoint. The query module
 * caches this ordered name list once per semantic filter, then each infinite
 * page hydrates only its own fixed-size slice.
 */
const listPaletteDateBoardImageNames = async ({
  boardId,
  createdFrom,
  createdTo,
  galleryView,
  orderDir,
  searchTerm,
  signal,
  starredFirst,
}: {
  boardId: string;
  createdFrom?: string;
  createdTo?: string;
  galleryView: GalleryView;
  orderDir: GalleryOrderDir;
  searchTerm: string;
  signal?: AbortSignal;
  starredFirst: boolean;
}): Promise<PaletteDateBoardImageNames> => {
  // Palette results remain intentionally image-only, but derive from the
  // polymorphic item_names endpoint so no webv2 path regresses to image_names.
  const result = await listGalleryDateBoardItemNames({
    boardId,
    createdFrom,
    createdTo,
    galleryView,
    orderDir,
    searchTerm,
    signal,
    starredFirst,
  });
  const imageNames = result.items.filter((ref) => ref.kind === 'image').map((ref) => ref.name);

  return {
    imageNames,
    total: imageNames.length,
  };
};

export interface GalleryItemNames {
  items: GalleryItemRef[];
  starredCount: number;
  total: number;
}

interface GalleryItemNamesRequest {
  boardId: string;
  createdFrom?: string;
  createdTo?: string;
  galleryView: GalleryView;
  orderDir: GalleryOrderDir;
  searchTerm: string;
  signal?: AbortSignal;
  starredFirst: boolean;
}

const mapGalleryItemNames = (body: {
  items: GalleryItemRef[];
  starred_count: number;
  total_count: number;
}): GalleryItemNames => ({
  items: body.items,
  starredCount: normalizeTotal(body.starred_count, 0),
  total: normalizeTotal(body.total_count, body.items.length),
});

export const listGalleryItemNames = async ({
  boardId,
  createdFrom,
  createdTo,
  galleryView,
  orderDir,
  searchTerm,
  signal,
  starredFirst,
}: GalleryItemNamesRequest): Promise<GalleryItemNames> => {
  const query = toSearchParams({
    board_id: boardId,
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    created_from: createdFrom,
    created_to: createdTo,
    is_intermediate: false,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred_first: starredFirst,
  });
  const body = await apiFetchJson<{
    items: GalleryItemRef[];
    starred_count: number;
    total_count: number;
  }>(`/api/v1/gallery/items/names?${query}`, { signal });

  return mapGalleryItemNames(body);
};

export const listGalleryDateBoardItemNames = async ({
  boardId,
  createdFrom,
  createdTo,
  galleryView,
  orderDir,
  searchTerm,
  signal,
  starredFirst,
}: GalleryItemNamesRequest): Promise<GalleryItemNames> => {
  if (
    (createdFrom !== undefined || createdTo !== undefined) &&
    !isTimestampInRange(getDateFromBoardId(boardId), { from: createdFrom, to: createdTo })
  ) {
    return { items: [], starredCount: 0, total: 0 };
  }

  const query = toSearchParams({
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred_first: starredFirst,
  });
  const body = await apiFetchJson<{
    items: GalleryItemRef[];
    starred_count: number;
    total_count: number;
  }>(`/api/v1/virtual_boards/by_date/${encodeURIComponent(getDateFromBoardId(boardId))}/item_names?${query}`, {
    signal,
  });

  return mapGalleryItemNames(body);
};

const hydrateVideoRefs = async (
  refs: readonly GalleryItemRef[],
  signal?: AbortSignal
): Promise<Map<number, GalleryVideoItem>> => {
  const videos = new Map<number, GalleryVideoItem>();
  let nextIndex = 0;

  const worker = async (): Promise<void> => {
    while (nextIndex < refs.length) {
      const index = nextIndex;
      nextIndex += 1;
      const ref = refs[index];

      if (!ref || ref.kind !== 'video') {
        continue;
      }

      try {
        videos.set(index, await getGalleryVideoByName(ref.name, signal));
      } catch (error: unknown) {
        if (!(error instanceof ApiError && error.status === 404)) {
          throw error;
        }
      }
    }
  };

  await Promise.all(Array.from({ length: Math.min(6, refs.length) }, () => worker()));

  return videos;
};

export const hydrateGalleryDateBoardItemPage = async ({
  items,
  limit,
  offset,
  signal,
  total,
}: Pick<GalleryItemNames, 'items' | 'total'> & {
  limit: number;
  offset: number;
  signal?: AbortSignal;
}): Promise<GalleryItemsPage> => {
  const refs = items.slice(offset, offset + limit);
  const imageNames = refs.filter((ref) => ref.kind === 'image').map((ref) => ref.name);
  const [images, videosByIndex] = await Promise.all([
    getGalleryImageItemsByNames(imageNames, signal),
    hydrateVideoRefs(refs, signal),
  ]);
  const imagesByName = new Map(images.map((image) => [image.name, image]));
  const hydrated = refs.flatMap((ref, index) => {
    const item = ref.kind === 'image' ? imagesByName.get(ref.name) : videosByIndex.get(index);

    return item ? [item] : [];
  });

  return { items: hydrated, total };
};

const hydratePaletteDateBoardImagePage = async ({
  imageNames,
  limit,
  offset,
  signal,
  total,
}: PaletteDateBoardImageNames & {
  limit: number;
  offset: number;
  signal?: AbortSignal;
}): Promise<GalleryImagesPage> => ({
  images: await getGalleryImagesByNames(imageNames.slice(offset, offset + limit), signal),
  total,
});

interface GalleryListRequest {
  boardId: string;
  createdFrom?: string;
  createdTo?: string;
  galleryView: GalleryView;
  limit?: number;
  offset?: number;
  orderDir?: GalleryOrderDir;
  searchTerm: string;
  signal?: AbortSignal;
  starredFirst?: boolean;
}

interface GalleryItemsRequest extends GalleryListRequest {
  isIntermediate?: boolean;
}

export const listGalleryItems = async ({
  boardId,
  createdFrom,
  createdTo,
  galleryView,
  isIntermediate = false,
  limit = 100,
  offset = 0,
  orderDir = 'DESC',
  searchTerm,
  signal,
  starredFirst = false,
}: GalleryItemsRequest): Promise<GalleryItemsPage> => {
  const query = toSearchParams({
    board_id: boardId,
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    created_from: createdFrom,
    created_to: createdTo,
    is_intermediate: isIntermediate,
    limit,
    offset,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred_first: starredFirst,
  });
  const body = await apiFetchJson<{
    items: BackendGalleryItemDTO[];
    limit: number;
    offset: number;
    total: number;
  }>(`/api/v1/gallery/items/?${query}`, { signal });

  return {
    items: body.items.map(mapGalleryItem),
    total: normalizeTotal(body.total, offset + body.items.length),
  };
};

export const listPaletteImages = async ({
  boardId,
  createdFrom,
  createdTo,
  galleryView,
  limit = 100,
  offset = 0,
  orderDir = 'DESC',
  searchTerm,
  signal,
  starredFirst = false,
}: GalleryListRequest): Promise<GalleryImagesPage> => {
  if (isDateBoardId(boardId)) {
    const names = await listPaletteDateBoardImageNames({
      boardId,
      createdFrom,
      createdTo,
      galleryView,
      orderDir,
      searchTerm,
      signal,
      starredFirst,
    });

    return hydratePaletteDateBoardImagePage({ ...names, limit, offset, signal });
  }

  const query = toSearchParams({
    board_id: boardId,
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    created_from: createdFrom,
    created_to: createdTo,
    is_intermediate: false,
    limit,
    offset,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred_first: starredFirst,
  });
  const body = await apiFetchJson<ListImagesResponse | BackendImageDTO[]>(`/api/v1/images/?${query}`, { signal });
  const items = Array.isArray(body) ? body : (body.items ?? []);

  return {
    images: items.map(mapImage),
    total: normalizeTotal(
      Array.isArray(body) ? undefined : body.total,
      offset + items.length + (Array.isArray(body) && items.length >= limit ? 1 : 0)
    ),
  };
};

/**
 * The `ImageRecordChanges` body that promotes a staged canvas candidate (an
 * intermediate image) into a durable, gallery-visible image: clearing
 * `is_intermediate` stops it being garbage-collected, and `image_category:
 * 'general'` surfaces it in the gallery's images view. Pure so the request
 * shape can be unit-tested without a fetch.
 */
export const imageSaveToGalleryChanges = (): { is_intermediate: false; image_category: 'general' } => ({
  image_category: 'general',
  is_intermediate: false,
});

/**
 * The `ImageRecordChanges` body that makes an intermediate image durable without
 * changing its category. Clearing `is_intermediate` stops garbage collection;
 * where the image appears is determined by its existing category.
 *
 * NOT for adopting a graph result as canvas pixels — a node's output is
 * `general`, so this alone publishes it to the gallery's Images view. Use
 * {@link imageMakeCanvasAssetChanges} there.
 */
export const imageMakeDurableChanges = (): { is_intermediate: false } => ({
  is_intermediate: false,
});

/**
 * The `ImageRecordChanges` body that adopts a utility result as CANVAS-OWNED
 * pixels: durable (so it is not garbage-collected out from under the layer that
 * now points at it) and `image_category: 'other'`, the category the canvas
 * already uploads its own paint bitmaps under.
 *
 * A node's output is `general` by default, which is exactly what the gallery's
 * Images view lists — so promoting a control-layer filter result with
 * {@link imageMakeDurableChanges} alone published every ControlNet preprocess
 * into the user's gallery. These are layer pixels, not gallery images, and
 * `'other'` belongs to neither {@link imageCategories} nor
 * {@link assetCategories}, so they surface in neither view.
 */
export const imageMakeCanvasAssetChanges = (): { is_intermediate: false; image_category: 'other' } => ({
  image_category: 'other',
  is_intermediate: false,
});

/**
 * Makes a single intermediate image durable (survives GC) without changing its
 * category, via `PATCH /api/v1/images/i/{image_name}`. Resolves once the PATCH
 * succeeds; the caller commits the layer-source swap only after this settles so a
 * failed PATCH never strands the layer pointing at a soon-to-be-collected image.
 */
export const makeImageDurable = async (imageName: string): Promise<void> => {
  await apiFetchJson<BackendImageDTO>(`/api/v1/images/i/${encodeURIComponent(imageName)}`, {
    body: JSON.stringify(imageMakeDurableChanges()),
    method: 'PATCH',
  });
};

/**
 * Adopts a utility result as canvas-owned pixels — durable AND out of the
 * gallery's Images view. Used wherever a graph result becomes a layer's
 * persisted source; {@link makeImageDurable} remains for callers that must keep
 * whatever category the image already had.
 */
export const makeImageCanvasAsset = async (imageName: string): Promise<void> => {
  await apiFetchJson<BackendImageDTO>(`/api/v1/images/i/${encodeURIComponent(imageName)}`, {
    body: JSON.stringify(imageMakeCanvasAssetChanges()),
    method: 'PATCH',
  });
};

/**
 * Promotes a single image (e.g. a staged canvas candidate) into the gallery via
 * `PATCH /api/v1/images/i/{image_name}` and returns the updated {@link GalleryImage}.
 */
export const saveImageToGallery = async (imageName: string): Promise<GalleryImage> => {
  const body = await apiFetchJson<BackendImageDTO>(`/api/v1/images/i/${encodeURIComponent(imageName)}`, {
    body: JSON.stringify(imageSaveToGalleryChanges()),
    method: 'PATCH',
  });

  return mapImage(body);
};

export const getGalleryImageMetadata = async (
  imageName: string,
  signal?: AbortSignal
): Promise<GalleryImageMetadata | null> => {
  const body = await apiFetchJson<unknown>(`/api/v1/images/i/${encodeURIComponent(imageName)}/metadata`, { signal });

  return body && typeof body === 'object' && !Array.isArray(body) ? (body as GalleryImageMetadata) : null;
};

export const createGalleryBoard = async (boardName: string, signal?: AbortSignal): Promise<GalleryBoard> => {
  const query = toSearchParams({ board_name: boardName });
  const body = await apiFetchJson<BackendBoardDTO>(`/api/v1/boards/?${query}`, { method: 'POST', signal });

  return mapBoard(body);
};

export const updateGalleryBoard = async (
  boardId: string,
  changes: { name?: string; archived?: boolean },
  signal?: AbortSignal
): Promise<GalleryBoard> => {
  const body = await apiFetchJson<BackendBoardDTO>(`/api/v1/boards/${encodeURIComponent(boardId)}`, {
    body: JSON.stringify({ archived: changes.archived, board_name: changes.name }),
    method: 'PATCH',
    signal,
  });

  return mapBoard(body);
};

export const deleteGalleryBoard = async (
  boardId: string,
  includeImages: boolean,
  signal?: AbortSignal
): Promise<GalleryBoardDeletionResult> => {
  const query = toSearchParams({ include_images: includeImages });
  const body = await apiFetchJson<{
    board_id: string;
    deleted_board_images: string[];
    deleted_board_videos?: string[];
    deleted_images: string[];
    deleted_videos?: string[];
    failed_images?: string[];
    failed_videos?: string[];
  }>(`/api/v1/boards/${encodeURIComponent(boardId)}?${query}`, { method: 'DELETE', signal });

  return {
    boardId: body.board_id,
    deletedBoardImageNames: body.deleted_board_images,
    deletedBoardVideoNames: body.deleted_board_videos ?? [],
    deletedImageNames: body.deleted_images,
    deletedVideoNames: body.deleted_videos ?? [],
    failedImageNames: body.failed_images ?? [],
    failedVideoNames: body.failed_videos ?? [],
  };
};

export interface GalleryItemOrganizationTransportResult {
  affectedBoardIds: string[];
  succeededNames: string[];
}

interface GalleryImageDeleteTransportResult extends GalleryItemOrganizationTransportResult {
  failedNames: string[];
}

const getRequiredStringArray = (body: unknown, field: string): string[] => {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new TypeError(`Gallery mutation response must be an object with "${field}".`);
  }

  const value = Reflect.get(body, field);

  if (!Array.isArray(value) || !value.every((item): item is string => typeof item === 'string' && item.length > 0)) {
    throw new TypeError(`Gallery mutation response field "${field}" must be an array of non-empty strings.`);
  }

  return value;
};

const getOptionalStringArray = (body: unknown, field: string): string[] => {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new TypeError(`Gallery mutation response must be an object.`);
  }

  const value = Reflect.get(body, field);

  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value) || !value.every((item): item is string => typeof item === 'string' && item.length > 0)) {
    throw new TypeError(`Gallery mutation response field "${field}" must be an array of non-empty strings.`);
  }

  return value;
};

const mapGalleryItemOrganizationTransportResult = (
  body: unknown,
  succeededField: string
): GalleryItemOrganizationTransportResult => ({
  affectedBoardIds: getRequiredStringArray(body, 'affected_boards'),
  succeededNames: getRequiredStringArray(body, succeededField),
});

const mapGalleryVideoOrganizationTransportResult = (
  body: unknown,
  succeededField: string
): GalleryItemOrganizationTransportResult => {
  const result = mapGalleryItemOrganizationTransportResult(body, succeededField);

  getRequiredStringArray(body, 'failed_videos');

  return result;
};

const emptyGalleryItemOrganizationTransportResult = (): GalleryItemOrganizationTransportResult => ({
  affectedBoardIds: [],
  succeededNames: [],
});

const isInvalidGalleryBoardDestination = (boardId: string): boolean =>
  boardId === 'generated' || boardId === 'assets' || isDateBoardId(boardId);

export const addGalleryImageItemsToBoard = async (
  boardId: string,
  imageNames: string[],
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> => {
  if (boardId === 'none' || isInvalidGalleryBoardDestination(boardId) || imageNames.length === 0) {
    return emptyGalleryItemOrganizationTransportResult();
  }

  signal?.throwIfAborted();
  const body = await apiFetchJson<unknown>('/api/v1/board_images/batch', {
    body: JSON.stringify({ board_id: boardId, image_names: imageNames }),
    method: 'POST',
    signal,
  });
  signal?.throwIfAborted();

  return mapGalleryItemOrganizationTransportResult(body, 'added_images');
};

export const addImagesToGalleryBoard = async (
  boardId: string,
  imageNames: string[],
  signal?: AbortSignal
): Promise<string[]> => (await addGalleryImageItemsToBoard(boardId, imageNames, signal)).succeededNames;

export const removeGalleryImageItemsFromBoard = async (
  imageNames: string[],
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> => {
  if (imageNames.length === 0) {
    return emptyGalleryItemOrganizationTransportResult();
  }

  signal?.throwIfAborted();
  const body = await apiFetchJson<unknown>('/api/v1/board_images/batch/delete', {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });
  signal?.throwIfAborted();

  return mapGalleryItemOrganizationTransportResult(body, 'removed_images');
};

export const removeImagesFromGalleryBoard = async (imageNames: string[], signal?: AbortSignal): Promise<string[]> =>
  (await removeGalleryImageItemsFromBoard(imageNames, signal)).succeededNames;

export const setGalleryImageItemsStarred = async (
  imageNames: string[],
  starred: boolean,
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> => {
  if (imageNames.length === 0) {
    return emptyGalleryItemOrganizationTransportResult();
  }

  signal?.throwIfAborted();
  const body = await apiFetchJson<unknown>(`/api/v1/images/${starred ? 'star' : 'unstar'}`, {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });
  signal?.throwIfAborted();

  return mapGalleryItemOrganizationTransportResult(body, starred ? 'starred_images' : 'unstarred_images');
};

export const starGalleryImages = (imageNames: string[], signal?: AbortSignal): Promise<string[]> =>
  setGalleryImageItemsStarred(imageNames, true, signal).then((result) => result.succeededNames);

export const unstarGalleryImages = (imageNames: string[], signal?: AbortSignal): Promise<string[]> =>
  setGalleryImageItemsStarred(imageNames, false, signal).then((result) => result.succeededNames);

export const deleteGalleryImageItems = async (
  imageNames: string[],
  signal?: AbortSignal
): Promise<GalleryImageDeleteTransportResult> => {
  if (imageNames.length === 0) {
    return { ...emptyGalleryItemOrganizationTransportResult(), failedNames: [] };
  }

  signal?.throwIfAborted();
  const body = await apiFetchJson<unknown>('/api/v1/images/delete', {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });
  signal?.throwIfAborted();

  return {
    ...mapGalleryItemOrganizationTransportResult(body, 'deleted_images'),
    failedNames: getOptionalStringArray(body, 'failed_images'),
  };
};

/**
 * Deletes images and reports the outcome per name.
 *
 * The backend deletes each name independently and no longer aborts the batch on the first
 * failure, so a request can partly succeed. Callers must evict only `deletedImageNames` from
 * their caches — treating the whole request as successful would hide a still-present image
 * until the next full refresh.
 */
export const deleteGalleryImages = async (
  imageNames: string[],
  signal?: AbortSignal
): Promise<GalleryDeletionResult> => {
  const result = await deleteGalleryImageItems(imageNames, signal);

  return {
    deletedImageNames: result.succeededNames,
    failedImageNames: result.failedNames,
  };
};

const mutateGalleryVideoItems = async (
  videoNames: string[],
  operation: 'delete' | 'star' | 'unstar',
  succeededField: 'deleted_videos' | 'starred_videos' | 'unstarred_videos',
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> => {
  if (videoNames.length === 0) {
    return emptyGalleryItemOrganizationTransportResult();
  }

  signal?.throwIfAborted();
  const body = await apiFetchJson<unknown>(`/api/v1/videos/${operation}`, {
    body: JSON.stringify({ video_names: videoNames }),
    method: 'POST',
    signal,
  });
  signal?.throwIfAborted();

  return mapGalleryVideoOrganizationTransportResult(body, succeededField);
};

export const deleteGalleryVideoItems = (
  videoNames: string[],
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> =>
  mutateGalleryVideoItems(videoNames, 'delete', 'deleted_videos', signal);

export const setGalleryVideoItemsStarred = (
  videoNames: string[],
  starred: boolean,
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> =>
  mutateGalleryVideoItems(
    videoNames,
    starred ? 'star' : 'unstar',
    starred ? 'starred_videos' : 'unstarred_videos',
    signal
  );

const moveGalleryVideoItemToBoard = async (
  videoName: string,
  boardId: string,
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> => {
  signal?.throwIfAborted();
  const removing = boardId === 'none';
  const body = await apiFetchJson<unknown>('/api/v1/videos/board', {
    body: JSON.stringify(removing ? { video_name: videoName } : { board_id: boardId, video_name: videoName }),
    method: removing ? 'DELETE' : 'POST',
    signal,
  });
  signal?.throwIfAborted();

  return mapGalleryItemOrganizationTransportResult(body, removing ? 'removed_videos' : 'added_videos');
};

const isFatalGalleryVideoMoveError = (error: unknown, signal: AbortSignal): boolean =>
  signal.aborted ||
  error instanceof AccountScopeExpiredError ||
  error instanceof HttpRequestIdentityExpiredError ||
  (error instanceof ApiError && error.status === 401) ||
  (error instanceof Error && error.name === 'AbortError');

export const moveGalleryVideoItemsToBoard = async (
  videoNames: string[],
  boardId: string,
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> => {
  if (videoNames.length === 0 || isInvalidGalleryBoardDestination(boardId)) {
    return emptyGalleryItemOrganizationTransportResult();
  }

  const owner = captureAccountScope();
  const requestSignal = signal ? AbortSignal.any([signal, owner.signal]) : owner.signal;
  const outcomes: (GalleryItemOrganizationTransportResult | undefined)[] = Array.from({
    length: videoNames.length,
  });
  const fatalFailure: { error: unknown; occurred: boolean } = { error: undefined, occurred: false };
  let nextIndex = 0;

  const claimNextVideo = (): { index: number; videoName: string } | null => {
    if (fatalFailure.occurred || requestSignal.aborted) {
      return null;
    }

    const index = nextIndex;
    const videoName = videoNames[index];

    if (videoName === undefined) {
      return null;
    }
    nextIndex += 1;

    return { index, videoName };
  };

  const worker = async (): Promise<void> => {
    while (true) {
      const claim = claimNextVideo();

      if (!claim) {
        return;
      }

      try {
        const outcome = await moveGalleryVideoItemToBoard(claim.videoName, boardId, requestSignal);

        assertAccountScopeCurrent(owner);
        requestSignal.throwIfAborted();
        if (fatalFailure.occurred) {
          return;
        }
        outcomes[claim.index] = outcome;
      } catch (error: unknown) {
        if (isFatalGalleryVideoMoveError(error, requestSignal)) {
          if (!fatalFailure.occurred) {
            fatalFailure.error = requestSignal.aborted ? (requestSignal.reason ?? error) : error;
            fatalFailure.occurred = true;
          }
          return;
        }
        // A rejected single-video request is unconfirmed. Other videos may still
        // return authoritative successes.
      }
    }
  };

  await Promise.all(Array.from({ length: Math.min(4, videoNames.length) }, () => worker()));

  if (fatalFailure.occurred) {
    throw fatalFailure.error;
  }
  assertAccountScopeCurrent(owner);
  requestSignal.throwIfAborted();

  const affectedBoardIds: string[] = [];
  const succeededNames: string[] = [];

  for (const [index, outcome] of outcomes.entries()) {
    const videoName = videoNames[index];

    if (!videoName || !outcome?.succeededNames.includes(videoName)) {
      continue;
    }
    succeededNames.push(videoName);
    affectedBoardIds.push(...outcome.affectedBoardIds);
  }

  return { affectedBoardIds, succeededNames };
};

const BULK_DOWNLOAD_POLL_INTERVAL_MS = 2000;
const BULK_DOWNLOAD_TIMEOUT_MS = 5 * 60 * 1000;

/**
 * Starts a bulk download (a zip prepared in a backend background task) and
 * polls the artifact endpoint until it exists. Returns the archive blob and
 * its file name.
 */
export const downloadGalleryArchive = async ({
  boardId,
  imageNames,
  signal,
}: {
  boardId?: string;
  imageNames?: string[];
  signal?: AbortSignal;
}): Promise<{ blob: Blob; fileName: string }> => {
  const owner = captureAccountScope();
  const requestSignal = signal ? AbortSignal.any([signal, owner.signal]) : owner.signal;
  const { bulk_download_item_name: fileName } = await apiFetchJson<{ bulk_download_item_name?: string | null }>(
    '/api/v1/images/download',
    {
      body: JSON.stringify({ board_id: boardId, image_names: imageNames }),
      method: 'POST',
      signal: requestSignal,
    }
  );

  assertAccountScopeCurrent(owner);
  requestSignal.throwIfAborted();
  if (!fileName) {
    throw new Error('The bulk download failed to start.');
  }

  const startedAt = Date.now();

  while (Date.now() - startedAt < BULK_DOWNLOAD_TIMEOUT_MS) {
    const response = await apiFetchRaw(`/api/v1/images/download/${encodeURIComponent(fileName)}`, {
      signal: requestSignal,
    });

    assertAccountScopeCurrent(owner);
    requestSignal.throwIfAborted();
    if (response.ok) {
      const blob = await response.blob();

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();
      return { blob, fileName };
    }

    if (response.status !== 404) {
      throw new Error(`${response.status} ${response.statusText}`);
    }

    await sleep(BULK_DOWNLOAD_POLL_INTERVAL_MS, requestSignal);
    assertAccountScopeCurrent(owner);
    requestSignal.throwIfAborted();
  }

  throw new Error('Timed out preparing the download archive.');
};

export const uploadGalleryImage = async (
  file: File,
  boardId: string,
  options: { isIntermediate?: boolean; signal?: AbortSignal } = {}
): Promise<GalleryImage> => {
  const query = toSearchParams({
    board_id: getUploadBoardId(boardId),
    image_category: 'user',
    is_intermediate: options.isIntermediate ?? false,
  });
  const body = new FormData();
  body.append('file', file);

  const uploadedImage = await apiFetchJson<BackendImageDTO>(`/api/v1/images/upload?${query}`, {
    body,
    method: 'POST',
    signal: options.signal,
  });

  return mapImage(uploadedImage);
};

export const uploadGalleryVideo = async (
  file: File,
  boardId: string,
  options: { signal?: AbortSignal } = {}
): Promise<GalleryVideoItem> => {
  const query = toSearchParams({
    board_id: getUploadBoardId(boardId),
    is_intermediate: false,
    video_category: 'general',
  });
  const body = new FormData();
  body.append('file', file);

  const uploadedVideo = await apiFetchJson<BackendVideoDTO>(`/api/v1/videos/upload?${query}`, {
    body,
    method: 'POST',
    signal: options.signal,
  });

  return mapVideo(uploadedVideo);
};

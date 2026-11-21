import type { components } from '@api/schema';
import type { ImageExportEncoding, ImageExportFormat, ImageExportSettings } from '@features/gallery/core/imageExport';
import type {
  GalleryImageItem,
  GalleryItem,
  GalleryItemCategory,
  GalleryItemKind,
  GalleryItemRef,
  GalleryItemsPage,
  GalleryVideoItem,
} from '@features/gallery/core/items';
import type { GallerySemanticQuery } from '@features/gallery/core/semanticImageQuery';
import type {
  GalleryBoard,
  GalleryGeneralAccess,
  GalleryBoardDeletionResult,
  GalleryBoardOrderBy,
  GalleryDeletionResult,
  GalleryImage,
  GalleryImageMetadata,
  GalleryImagesPage,
  GalleryOrderDir,
  GalleryView,
} from '@features/gallery/core/types';

import { getGalleryImageFullPath } from '@features/gallery/core/imagePaths';
import {
  ALL_READABLE_BOARDS_ID,
  DATE_BOARD_ID_PREFIX,
  getGalleryViewCategories,
  isDateBoardId,
  parseGalleryItemKey,
} from '@features/gallery/core/items';
import { getExternalImageFile, getImageCluster } from '@features/gallery/core/semanticImageQuery';
import { createLogger } from '@platform/logging/logger';
import { isTimestampInRange } from '@platform/search/dateTokens';
import {
  AccountScopeExpiredError,
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import {
  absolutizeApiUrl,
  ApiError,
  apiFetch,
  apiFetchJson,
  apiFetchRaw,
  assertOk,
  HttpRequestIdentityExpiredError,
  sleep,
} from '@platform/transport/http';

import { getGalleryImageThumbnailUrl } from './imageUrls';
import { getGalleryVideoThumbnailUrl } from './videoUrls';

type BackendBoardDTO = components['schemas']['BoardDTO'];

export { ALL_READABLE_BOARDS_ID, isDateBoardId };

/** The server's `MAX_IMAGE_BATCH_SIZE`/`MAX_VIDEO_BATCH_SIZE` (`api/routers/_limits.py`) refuse a longer list whole. */
export const GALLERY_REQUEST_MAX_NAMES = 1000;

const batchLogger = createLogger({ area: 'batch-requests', namespace: 'gallery' });

export const toNameBatches = (names: readonly string[]): string[][] => {
  const distinct = [...new Set(names)];
  const batches: string[][] = [];

  for (let offset = 0; offset < distinct.length; offset += GALLERY_REQUEST_MAX_NAMES) {
    batches.push(distinct.slice(offset, offset + GALLERY_REQUEST_MAX_NAMES));
  }

  return batches;
};

const getDateFromBoardId = (boardId: string): string => boardId.slice(DATE_BOARD_ID_PREFIX.length);

const getUploadBoardId = (boardId: string): string | undefined =>
  boardId === 'none' || boardId === ALL_READABLE_BOARDS_ID || isDateBoardId(boardId) ? undefined : boardId;

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
  has_workflow?: boolean;
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
  media_origin?: string | null;
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
  has_workflow?: boolean;
  height: number;
  is_intermediate: boolean;
  media_origin?: string | null;
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

const imageCategories = getGalleryViewCategories('images');
const assetCategories = getGalleryViewCategories('assets');

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

/** The backend sets exactly one of `cover_image_name` / `cover_video_name`; both resolve to a WebP thumbnail. */
const getBoardCoverThumbnailUrl = (board: {
  cover_image_name?: string | null;
  cover_video_name?: string | null;
}): string | undefined => {
  if (board.cover_image_name) {
    return getGalleryImageThumbnailUrl(board.cover_image_name);
  }

  return board.cover_video_name ? getGalleryVideoThumbnailUrl(board.cover_video_name) : undefined;
};

const mapBoard = (board: BackendBoardDTO): GalleryBoard => ({
  archived: board.archived,
  assetCount: board.asset_count,
  assetVideoCount: board.asset_video_count ?? 0,
  coverImageName: board.cover_image_name,
  coverThumbnailUrl: getBoardCoverThumbnailUrl(board),
  coverVideoName: board.cover_video_name,
  createdAt: board.created_at ?? null,
  id: board.board_id,
  imageCount: board.image_count,
  isInbox: board.is_inbox ?? false,
  kind: 'board',
  name: board.board_name,
  ownerName: board.owner_username ?? null,
  projectId: board.project_id ?? null,
  videoCount: board.video_count ?? 0,
  ...(board.allowed_actions === undefined
    ? {}
    : {
        access: {
          actions: board.allowed_actions,
          adminOverride: board.admin_override ?? false,
          generalAccess: board.general_access ?? 'restricted',
          role: board.my_role ?? null,
        },
      }),
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

const getGalleryVideoTotal = async ({
  boardId,
  categories,
  signal,
}: {
  boardId: string;
  categories?: string[];
  signal?: AbortSignal;
}): Promise<number> => {
  const query = toSearchParams({ board_id: boardId, categories, is_intermediate: false, limit: 0, offset: 0 });
  const body = await apiFetchJson<{ total: number }>(`/api/v1/videos/?${query}`, { signal });

  return body.total;
};

const mapImage = (image: BackendImageDTO): GalleryImage => ({
  boardId: image.board_id ?? 'none',
  createdAt: image.created_at,
  hasWorkflow: image.has_workflow,
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

const mediaOriginOf = (value: string | null | undefined): { mediaOrigin?: string } =>
  typeof value === 'string' && value ? { mediaOrigin: value } : {};

const mapGalleryItemBase = (
  item: BackendGalleryItemDTO
): Omit<GalleryItem, 'durationSeconds' | 'fps' | 'kind' | 'mediaOrigin' | 'sourceQueueItemId'> => ({
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
    ...mediaOriginOf(item.media_origin),
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
    ...(video.has_workflow === undefined ? {} : { hasWorkflow: video.has_workflow }),
    height: video.height,
    isIntermediate: video.is_intermediate,
    kind: 'video',
    ...mediaOriginOf(video.media_origin),
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

  const [
    body,
    uncategorizedImageCount,
    uncategorizedAssetCount,
    uncategorizedVideoCount,
    uncategorizedAssetVideoCount,
  ] = await Promise.all([
    boardsBodyPromise,
    getGalleryTotal({ boardId: 'none', categories: imageCategories, signal }),
    getGalleryTotal({ boardId: 'none', categories: assetCategories, signal }),
    // The uncategorized pseudo-board is assembled here, so its video totals need their own requests.
    getGalleryVideoTotal({ boardId: 'none', categories: [...imageCategories, ...assetCategories], signal }),
    getGalleryVideoTotal({ boardId: 'none', categories: assetCategories, signal }),
  ]);
  const boards = Array.isArray(body) ? body : (body.items ?? []);

  return [
    {
      archived: false,
      assetCount: uncategorizedAssetCount,
      assetVideoCount: uncategorizedAssetVideoCount,
      id: 'none',
      imageCount: uncategorizedImageCount,
      isInbox: false,
      kind: 'uncategorized',
      // `kind` is the durable fact; the UI resolves the label, so no untranslatable name crosses the transport.
      name: '',
      projectId: null,
      videoCount: uncategorizedVideoCount,
    },
    ...boards.filter((board) => includeArchived || !board.archived).map(mapBoard),
  ];
};

/** Zero image_count does not imply an empty board: date boards can hold only videos. */
interface VirtualDateBoardDTO {
  virtual_board_id: string;
  board_name: string;
  date: string;
  image_count: number;
  asset_count: number;
  video_count?: number;
  asset_video_count?: number;
  cover_image_name?: string | null;
  cover_video_name?: string | null;
}

export const listGalleryDateBoards = async (signal?: AbortSignal): Promise<GalleryBoard[]> => {
  const body = await apiFetchJson<VirtualDateBoardDTO[]>('/api/v1/virtual_boards/by_date', { signal });

  return body.map((board) => ({
    archived: false,
    assetCount: board.asset_count,
    assetVideoCount: board.asset_video_count ?? 0,
    coverImageName: board.cover_image_name,
    coverThumbnailUrl: getBoardCoverThumbnailUrl(board),
    coverVideoName: board.cover_video_name,
    id: board.virtual_board_id,
    imageCount: board.image_count,
    isInbox: false,
    kind: 'date',
    name: board.board_name,
    projectId: null,
    videoCount: board.video_count ?? 0,
  }));
};

const fetchImagesByNames = async (imageNames: readonly string[], signal?: AbortSignal): Promise<BackendImageDTO[]> => {
  const owner = captureAccountScope();
  const images: BackendImageDTO[] = [];

  for (const batch of toNameBatches(imageNames)) {
    signal?.throwIfAborted();
    assertAccountScopeCurrent(owner);
    images.push(
      ...(await apiFetchJson<BackendImageDTO[]>('/api/v1/images/images_by_names', {
        body: JSON.stringify({ image_names: batch }),
        method: 'POST',
        signal,
      }))
    );
  }

  return images;
};

export const getGalleryImagesByNames = async (imageNames: string[], signal?: AbortSignal): Promise<GalleryImage[]> => {
  if (imageNames.length === 0) {
    return [];
  }

  const body = await fetchImagesByNames(imageNames, signal);
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

  const body = await fetchImagesByNames(imageNames, signal);
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

export interface GalleryMediaWorkflow {
  graph: string | null;
  resolved_inputs?: string | null;
  workflow: string | null;
}

export const getGalleryVideoWorkflow = (videoName: string, signal?: AbortSignal): Promise<GalleryMediaWorkflow> =>
  apiFetchJson<GalleryMediaWorkflow>(`/api/v1/videos/i/${encodeURIComponent(videoName)}/workflow`, { signal });

export const getGalleryImageWorkflow = (imageName: string, signal?: AbortSignal): Promise<GalleryMediaWorkflow> =>
  apiFetchJson<GalleryMediaWorkflow>(`/api/v1/images/i/${encodeURIComponent(imageName)}/workflow`, { signal });

interface PaletteDateBoardImageNames {
  imageNames: string[];
  total: number;
}

/** Date boards have no offset-paginated endpoint: the query module caches the names and pages hydrate slices. */
const listPaletteDateBoardImageNames = async ({
  boardId,
  createdFrom,
  createdTo,
  galleryView,
  orderDir,
  searchTerm,
  signal,
}: {
  boardId: string;
  createdFrom?: string;
  createdTo?: string;
  galleryView: GalleryView;
  orderDir: GalleryOrderDir;
  searchTerm: string;
  signal?: AbortSignal;
}): Promise<PaletteDateBoardImageNames> => {
  const result = await listGalleryDateBoardItemNames({
    boardId,
    createdFrom,
    createdTo,
    galleryView,
    mediaKind: 'image',
    orderDir,
    searchTerm,
    signal,
  });
  const imageNames = result.items.filter((ref) => ref.kind === 'image').map((ref) => ref.name);

  return {
    imageNames,
    total: imageNames.length,
  };
};

export interface GalleryItemNames {
  items: GalleryItemRef[];
  total: number;
}

export type GalleryItemLocation = components['schemas']['GalleryItemLocation'];

interface GalleryItemNamesRequest {
  boardId: string;
  createdFrom?: string;
  createdTo?: string;
  galleryView: GalleryView;
  mediaKind?: GalleryItemKind;
  orderDir: GalleryOrderDir;
  searchTerm: string;
  signal?: AbortSignal;
  starred?: boolean;
}

const mapGalleryItemNames = (body: { items: GalleryItemRef[]; total_count: number }): GalleryItemNames => ({
  items: body.items,
  total: normalizeTotal(body.total_count, body.items.length),
});

export const listGalleryItemNames = async ({
  boardId,
  createdFrom,
  createdTo,
  galleryView,
  mediaKind,
  orderDir,
  searchTerm,
  signal,
  starred,
}: GalleryItemNamesRequest): Promise<GalleryItemNames> => {
  const query = toSearchParams({
    board_id: boardId,
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    created_from: createdFrom,
    created_to: createdTo,
    is_intermediate: false,
    media_kind: mediaKind,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred,
    // The backend defaults to starred-first; the grid orders chronologically.
    starred_first: false,
  });
  const body = await apiFetchJson<{ items: GalleryItemRef[]; total_count: number }>(
    `/api/v1/gallery/items/names?${query}`,
    { signal }
  );

  return mapGalleryItemNames(body);
};

export const getGalleryItemLocation = ({
  boardId,
  createdFrom,
  createdTo,
  galleryView,
  kind,
  mediaKind,
  name,
  orderDir,
  searchTerm,
  signal,
  starred,
}: GalleryItemNamesRequest & GalleryItemRef): Promise<GalleryItemLocation> => {
  const query = toSearchParams({
    board_id: boardId,
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    created_from: createdFrom,
    created_to: createdTo,
    is_intermediate: false,
    kind,
    media_kind: mediaKind,
    order_dir: orderDir,
    name,
    search_term: searchTerm.trim() || undefined,
    starred,
    starred_first: false,
  });

  return apiFetchJson<GalleryItemLocation>(`/api/v1/gallery/items/location?${query}`, { signal });
};

export const listGalleryDateBoardItemNames = async ({
  boardId,
  createdFrom,
  createdTo,
  galleryView,
  mediaKind,
  orderDir,
  searchTerm,
  signal,
  starred,
}: GalleryItemNamesRequest): Promise<GalleryItemNames> => {
  if (
    (createdFrom !== undefined || createdTo !== undefined) &&
    !isTimestampInRange(getDateFromBoardId(boardId), { from: createdFrom, to: createdTo })
  ) {
    return { items: [], total: 0 };
  }

  const query = toSearchParams({
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    media_kind: mediaKind,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred,
    starred_first: false,
  });
  const body = await apiFetchJson<{ items: GalleryItemRef[]; total_count: number }>(
    `/api/v1/virtual_boards/by_date/${encodeURIComponent(getDateFromBoardId(boardId))}/item_names?${query}`,
    { signal }
  );

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
  const hydrated: GalleryItem[] = [];
  const itemIndices: number[] = [];

  refs.forEach((ref, index) => {
    const item = ref.kind === 'image' ? imagesByName.get(ref.name) : videosByIndex.get(index);

    if (item) {
      hydrated.push(item);
      itemIndices.push(offset + index);
    }
  });

  return { items: hydrated, itemIndices, offset, total };
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
  /** Ignored by the image-only palette listing. */
  mediaKind?: GalleryItemKind;
  offset?: number;
  orderDir?: GalleryOrderDir;
  searchTerm: string;
  signal?: AbortSignal;
  starred?: boolean;
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
  mediaKind,
  offset = 0,
  orderDir = 'DESC',
  searchTerm,
  signal,
  starred,
}: GalleryItemsRequest): Promise<GalleryItemsPage> => {
  const query = toSearchParams({
    board_id: boardId,
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    created_from: createdFrom,
    created_to: createdTo,
    is_intermediate: isIntermediate,
    limit,
    media_kind: mediaKind,
    offset,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred,
    starred_first: false,
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

export const SEMANTIC_SEARCH_MAX_RESULTS = 500;

export interface GallerySemanticResult {
  ref: GalleryItemRef;
  score: number;
}

type SemanticSearchBody = { results: { image_name: string; kind?: string; score: number }[] };

const toSemanticResults = (body: SemanticSearchBody): GallerySemanticResult[] =>
  // `image_name` serves both namespaces; only kind `video` selects videos.
  body.results.map((result) => ({
    ref: { kind: result.kind === 'video' ? 'video' : 'image', name: result.image_name },
    score: result.score,
  }));

const toSemanticScopeParams = (boardId: string | undefined): { board_id?: string; created_date?: string } => {
  if (boardId === undefined || boardId === ALL_READABLE_BOARDS_ID) {
    return {};
  }

  return isDateBoardId(boardId) ? { created_date: getDateFromBoardId(boardId) } : { board_id: boardId };
};

/** A set `mediaKind` overrides include_videos server-side. */
export const searchGallerySemantic = async (
  query: Exclude<GallerySemanticQuery, { kind: 'cluster' }>,
  {
    boardId,
    includeVideos = true,
    limit = SEMANTIC_SEARCH_MAX_RESULTS,
    mediaKind,
    signal,
  }: {
    boardId?: string;
    includeVideos?: boolean;
    limit?: number;
    mediaKind?: GalleryItemKind;
    signal?: AbortSignal;
  } = {}
): Promise<GallerySemanticResult[]> => {
  const scope = { ...toSemanticScopeParams(boardId), include_videos: includeVideos, limit, media_kind: mediaKind };

  if (query.kind === 'url') {
    const params = toSearchParams({ image_url: query.url, ...scope });

    return toSemanticResults(
      await apiFetchJson<SemanticSearchBody>(`/api/v1/image_map/search_by_image?${params}`, {
        method: 'POST',
        signal,
      })
    );
  }

  if (query.kind === 'file') {
    const entry = getExternalImageFile(query.fileId);

    if (!entry) {
      throw new Error('The dropped image is no longer available; drop it again to search.');
    }

    const form = new FormData();

    form.append('image', entry.blob, entry.label || 'image');

    return toSemanticResults(
      await apiFetchJson<SemanticSearchBody>(`/api/v1/image_map/search_by_image?${toSearchParams(scope)}`, {
        body: form,
        method: 'POST',
        signal,
      })
    );
  }

  const params = toSearchParams(
    query.kind === 'text' ? { ...scope, q: query.query } : { image_name: query.imageName, ...scope }
  );

  return toSemanticResults(await apiFetchJson<SemanticSearchBody>(`/api/v1/image_map/search?${params}`, { signal }));
};

export interface ImageIndexAvailability {
  /** `switching`: an installed replacement model starts once the retired one's work drains. */
  state: 'disabled' | 'model_missing' | 'switching' | 'ready';
  /** Set only while it is missing. */
  modelName: string | null;
}

interface ImageIndexStatusBody {
  enabled: boolean;
  model_name?: string | null;
  projection?: { state?: string };
}

/** Search needs embeddings and their model, not map projection; an inactive computing index is switching models. */
export const fetchImageIndexAvailability = async (signal: AbortSignal): Promise<ImageIndexAvailability> => {
  const body = await apiFetchJson<ImageIndexStatusBody>('/api/v1/image_map/status', { signal });

  if (body.enabled) {
    return { modelName: null, state: 'ready' };
  }
  if (body.model_name) {
    return { modelName: body.model_name, state: 'model_missing' };
  }

  return { modelName: null, state: body.projection?.state === 'computing' ? 'switching' : 'disabled' };
};

/** Kind-qualified refs keep relevance order for hydration, range selection and deletion neighbors. */
export const listSemanticGalleryItemNames = async ({
  boardId,
  mediaKind,
  query,
  signal,
}: {
  boardId?: string;
  mediaKind?: GalleryItemKind;
  query: GallerySemanticQuery;
  signal?: AbortSignal;
}): Promise<GalleryItemNames> => {
  // A cluster query is a client-held member list; there is nothing to ask the server.
  if (query.kind === 'cluster') {
    const items = (getImageCluster(query.clusterId)?.itemKeys ?? [])
      .map(parseGalleryItemKey)
      .filter((ref) => mediaKind === undefined || ref.kind === mediaKind);

    return { items, total: items.length };
  }

  const results = await searchGallerySemantic(query, { boardId, mediaKind, signal });

  return {
    items: results.map((result) => result.ref),
    total: results.length,
  };
};

export const listPaletteSemanticImages = async ({
  limit,
  query,
  signal,
}: {
  limit: number;
  query: string;
  signal?: AbortSignal;
}): Promise<GalleryImage[]> => {
  const results = await searchGallerySemantic({ kind: 'text', query }, { includeVideos: false, limit, signal });

  return getGalleryImagesByNames(
    results.map((result) => result.ref.name),
    signal
  );
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
    starred_first: false,
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

export const imageSaveToGalleryChanges = (): { is_intermediate: false; image_category: 'general' } => ({
  image_category: 'general',
  is_intermediate: false,
});

/** Canvas adoption needs {@link imageMakeCanvasAssetChanges}, or general outputs publish. */
export const imageMakeDurableChanges = (): { is_intermediate: false } => ({
  is_intermediate: false,
});

/** Durable `other` pixels outside both gallery views. */
export const imageMakeCanvasAssetChanges = (): { is_intermediate: false; image_category: 'other' } => ({
  image_category: 'other',
  is_intermediate: false,
});

/** Await before a layer-source swap: a layer must never reference collectible pixels. */
export const makeImageDurable = async (imageName: string): Promise<void> => {
  await apiFetchJson<BackendImageDTO>(`/api/v1/images/i/${encodeURIComponent(imageName)}`, {
    body: JSON.stringify(imageMakeDurableChanges()),
    method: 'PATCH',
  });
};

/** {@link makeImageDurable} keeps the category instead. */
export const makeImageCanvasAsset = async (imageName: string): Promise<void> => {
  await apiFetchJson<BackendImageDTO>(`/api/v1/images/i/${encodeURIComponent(imageName)}`, {
    body: JSON.stringify(imageMakeCanvasAssetChanges()),
    method: 'PATCH',
  });
};

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

/** False only on a 404: a 403 board exists, so the project keeps naming it. Other failures throw. */
export const galleryBoardExists = async (boardId: string, signal?: AbortSignal): Promise<boolean> => {
  if (getUploadBoardId(boardId) === undefined) {
    return true;
  }

  try {
    await apiFetchJson<BackendBoardDTO>(`/api/v1/boards/${encodeURIComponent(boardId)}`, { signal });

    return true;
  } catch (error) {
    if (error instanceof ApiError && (error.status === 404 || error.status === 403)) {
      return error.status === 403;
    }

    throw error;
  }
};

/** A project must be the caller's own, else the backend answers 404. */
export const createGalleryBoard = async (
  boardName: string,
  projectId: string | null = null,
  signal?: AbortSignal
): Promise<GalleryBoard> => {
  const query = toSearchParams({ board_name: boardName, ...(projectId === null ? {} : { project_id: projectId }) });
  const body = await apiFetchJson<BackendBoardDTO>(`/api/v1/boards/?${query}`, { method: 'POST', signal });

  return mapBoard(body);
};

/** Null moves the board to the Library; undefined means unchanged. */
export const updateGalleryBoard = async (
  boardId: string,
  changes: { name?: string; archived?: boolean; generalAccess?: GalleryGeneralAccess; projectId?: string | null },
  signal?: AbortSignal
): Promise<GalleryBoard> => {
  const body = await apiFetchJson<BackendBoardDTO>(`/api/v1/boards/${encodeURIComponent(boardId)}`, {
    // JSON.stringify drops undefined keys, leaving a field unchanged; an explicit null must survive.
    body: JSON.stringify({
      archived: changes.archived,
      board_name: changes.name,
      general_access: changes.generalAccess,
      ...(changes.projectId !== undefined ? { project_id: changes.projectId } : {}),
    }),
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

interface GalleryNameBatchResult extends GalleryItemOrganizationTransportResult {
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

/**
 * One batch at a time. After earlier batches succeeded, a failure or abort reports them and marks the rest failed;
 * before that it rejects like a single request. An account change rejects whatever succeeded.
 */
const mutateInNameBatches = async (
  names: readonly string[],
  signal: AbortSignal | undefined,
  send: (batch: string[]) => Promise<GalleryItemOrganizationTransportResult & { failedNames?: string[] }>
): Promise<GalleryNameBatchResult> => {
  const owner = captureAccountScope();
  const batches = toNameBatches(names);
  const affectedBoardIds = new Set<string>();
  const failedNames: string[] = [];
  const succeededNames: string[] = [];

  for (const [index, batch] of batches.entries()) {
    try {
      signal?.throwIfAborted();
      assertAccountScopeCurrent(owner);
      const result = await send(batch);

      succeededNames.push(...result.succeededNames);
      failedNames.push(...(result.failedNames ?? []));
      result.affectedBoardIds.forEach((boardId) => affectedBoardIds.add(boardId));
    } catch (error: unknown) {
      if (succeededNames.length === 0 || !isAccountScopeCurrent(owner)) {
        throw error;
      }
      batchLogger.debug({
        context: { batch: index + 1, batches: batches.length, unconfirmed: batches.slice(index).flat().length },
        error,
        message: 'A gallery batch request failed after earlier batches succeeded',
        name: 'gallery.batch-request-failed',
      });
      failedNames.push(...batches.slice(index).flat());
      break;
    }
  }

  return { affectedBoardIds: [...affectedBoardIds], failedNames, succeededNames };
};

export const isInvalidGalleryBoardDestination = (boardId: string): boolean =>
  boardId === 'generated' || boardId === 'assets' || isDateBoardId(boardId);

export const addGalleryImageItemsToBoard = (
  boardId: string,
  imageNames: string[],
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> => {
  if (boardId === 'none' || isInvalidGalleryBoardDestination(boardId)) {
    return Promise.resolve(emptyGalleryItemOrganizationTransportResult());
  }

  return mutateInNameBatches(imageNames, signal, async (batch) => {
    const body = await apiFetchJson<unknown>('/api/v1/board_images/batch', {
      body: JSON.stringify({ board_id: boardId, image_names: batch }),
      method: 'POST',
      signal,
    });
    signal?.throwIfAborted();

    return mapGalleryItemOrganizationTransportResult(body, 'added_images');
  });
};

export const addImagesToGalleryBoard = async (
  boardId: string,
  imageNames: string[],
  signal?: AbortSignal
): Promise<string[]> => (await addGalleryImageItemsToBoard(boardId, imageNames, signal)).succeededNames;

export const removeGalleryImageItemsFromBoard = (
  imageNames: string[],
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> =>
  mutateInNameBatches(imageNames, signal, async (batch) => {
    const body = await apiFetchJson<unknown>('/api/v1/board_images/batch/delete', {
      body: JSON.stringify({ image_names: batch }),
      method: 'POST',
      signal,
    });
    signal?.throwIfAborted();

    return mapGalleryItemOrganizationTransportResult(body, 'removed_images');
  });

export const removeImagesFromGalleryBoard = async (imageNames: string[], signal?: AbortSignal): Promise<string[]> =>
  (await removeGalleryImageItemsFromBoard(imageNames, signal)).succeededNames;

export const setGalleryImageItemsStarred = (
  imageNames: string[],
  starred: boolean,
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> =>
  mutateInNameBatches(imageNames, signal, async (batch) => {
    const body = await apiFetchJson<unknown>(`/api/v1/images/${starred ? 'star' : 'unstar'}`, {
      body: JSON.stringify({ image_names: batch }),
      method: 'POST',
      signal,
    });
    signal?.throwIfAborted();

    return mapGalleryItemOrganizationTransportResult(body, starred ? 'starred_images' : 'unstarred_images');
  });

export const starGalleryImages = (imageNames: string[], signal?: AbortSignal): Promise<string[]> =>
  setGalleryImageItemsStarred(imageNames, true, signal).then((result) => result.succeededNames);

export const unstarGalleryImages = (imageNames: string[], signal?: AbortSignal): Promise<string[]> =>
  setGalleryImageItemsStarred(imageNames, false, signal).then((result) => result.succeededNames);

export const deleteGalleryImageItems = (imageNames: string[], signal?: AbortSignal): Promise<GalleryNameBatchResult> =>
  mutateInNameBatches(imageNames, signal, async (batch) => {
    const body = await apiFetchJson<unknown>('/api/v1/images/delete', {
      body: JSON.stringify({ image_names: batch }),
      method: 'POST',
      signal,
    });
    signal?.throwIfAborted();

    return {
      ...mapGalleryItemOrganizationTransportResult(body, 'deleted_images'),
      failedNames: getOptionalStringArray(body, 'failed_images'),
    };
  });

/** Evict only deletedImageNames; other names may still exist on the server. */
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

const mutateGalleryVideoItems = (
  videoNames: string[],
  operation: 'delete' | 'star' | 'unstar',
  succeededField: 'deleted_videos' | 'starred_videos' | 'unstarred_videos',
  signal?: AbortSignal
): Promise<GalleryItemOrganizationTransportResult> =>
  mutateInNameBatches(videoNames, signal, async (batch) => {
    const body = await apiFetchJson<unknown>(`/api/v1/videos/${operation}`, {
      body: JSON.stringify({ video_names: batch }),
      method: 'POST',
      signal,
    });
    signal?.throwIfAborted();

    return mapGalleryVideoOrganizationTransportResult(body, succeededField);
  });

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
        // A rejected single-video request is unconfirmed. Other videos may still return authoritative successes.
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
const BULK_EXPORT_TIMEOUT_PER_IMAGE_MS = 5000;

const toImageExportQuery = (encoding: ImageExportEncoding): string =>
  toSearchParams({
    format: encoding.format,
    long_edge: encoding.longEdge ?? undefined,
    metadata: encoding.metadata,
    quality: encoding.quality ?? undefined,
  });

export const exportGalleryImage = async (
  imageName: string,
  encoding: ImageExportEncoding,
  signal?: AbortSignal
): Promise<Blob> => {
  const owner = captureAccountScope();
  const response = await apiFetch(
    `/api/v1/images/i/${encodeURIComponent(imageName)}/export?${toImageExportQuery(encoding)}`,
    { signal: signal ? AbortSignal.any([signal, owner.signal]) : owner.signal }
  );
  const blob = await response.blob();

  assertAccountScopeCurrent(owner);
  return blob;
};

export const resolveGalleryExportFileNames = async (
  {
    format,
    imageNames,
    pattern,
  }: {
    format: ImageExportFormat;
    imageNames: readonly string[];
    pattern: string;
  },
  signal?: AbortSignal
): Promise<string[]> => {
  const owner = captureAccountScope();
  const { file_names: fileNames } = await apiFetchJson<{ file_names: string[] }>('/api/v1/images/export/names', {
    body: JSON.stringify({ format, image_names: imageNames, pattern }),
    method: 'POST',
    signal: signal ? AbortSignal.any([signal, owner.signal]) : owner.signal,
  });

  assertAccountScopeCurrent(owner);
  return fileNames;
};

export const getGalleryImageFileSize = async (imageName: string, signal?: AbortSignal): Promise<number | null> => {
  const owner = captureAccountScope();
  const response = await apiFetch(getGalleryImageFullPath(imageName), {
    method: 'HEAD',
    signal: signal ? AbortSignal.any([signal, owner.signal]) : owner.signal,
  });
  const length = Number(response.headers.get('content-length') ?? Number.NaN);

  assertAccountScopeCurrent(owner);
  return Number.isSafeInteger(length) && length > 0 ? length : null;
};

export const downloadGalleryArchive = async ({
  boardId,
  exportSettings,
  imageNames,
  signal,
}: {
  boardId?: string;
  exportSettings?: ImageExportSettings;
  imageNames?: string[];
  signal?: AbortSignal;
}): Promise<{ blob: Blob; fileName: string }> => {
  const owner = captureAccountScope();
  const requestSignal = signal ? AbortSignal.any([signal, owner.signal]) : owner.signal;
  const { bulk_download_item_name: fileName } = await apiFetchJson<{ bulk_download_item_name?: string | null }>(
    '/api/v1/images/download',
    {
      body: JSON.stringify({
        board_id: boardId,
        export: exportSettings
          ? {
              format: exportSettings.format,
              long_edge: exportSettings.longEdge,
              metadata: exportSettings.metadata,
              name_pattern: exportSettings.namePattern,
              quality: exportSettings.quality,
            }
          : undefined,
        image_names: imageNames,
      }),
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
  const timeoutMs =
    BULK_DOWNLOAD_TIMEOUT_MS + (exportSettings ? (imageNames?.length ?? 0) * BULK_EXPORT_TIMEOUT_PER_IMAGE_MS : 0);

  while (Date.now() - startedAt < timeoutMs) {
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

    // 404 means not ready yet; any other answer is the job's failure, with its reason in `detail`.
    if (response.status !== 404) {
      await assertOk(response);
    }

    await sleep(BULK_DOWNLOAD_POLL_INTERVAL_MS, requestSignal);
    assertAccountScopeCurrent(owner);
    requestSignal.throwIfAborted();
  }

  throw new Error(
    exportSettings ? 'Timed out preparing the export archive.' : 'Timed out preparing the download archive.'
  );
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
    // Uploads are user assets, like `image_category: 'user'`; generated videos save as 'general'.
    video_category: 'user',
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

import type {
  GalleryBoard,
  GalleryBoardOrderBy,
  GalleryDeletionResult,
  GalleryImage,
  GalleryImageMetadata,
  GalleryImagesPage,
  GalleryOrderDir,
  GalleryView,
} from '@features/gallery/core/types';

import { isTimestampInRange } from '@platform/search/dateTokens';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { absolutizeApiUrl, apiFetch, apiFetchJson, apiFetchRaw, sleep } from '@platform/transport/http';

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

export const getGalleryImageByName = async (imageName: string, signal?: AbortSignal): Promise<GalleryImage> => {
  const body = await apiFetchJson<BackendImageDTO>(`/api/v1/images/i/${encodeURIComponent(imageName)}`, { signal });

  return mapImage(body);
};

export interface GalleryDateBoardImageNames {
  imageNames: string[];
  total: number;
}

/**
 * Date virtual boards have no offset-paginated DTO endpoint. The query module
 * caches this ordered name list once per semantic filter, then each infinite
 * page hydrates only its own fixed-size slice.
 */
export const listGalleryDateBoardImageNames = async ({
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
}): Promise<GalleryDateBoardImageNames> => {
  // The board already pins a single day; a date filter either contains that
  // day (and constrains nothing further) or excludes the whole board.
  if (
    (createdFrom !== undefined || createdTo !== undefined) &&
    !isTimestampInRange(getDateFromBoardId(boardId), { from: createdFrom, to: createdTo })
  ) {
    return { imageNames: [], total: 0 };
  }

  const query = toSearchParams({
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred_first: starredFirst,
  });
  const body = await apiFetchJson<{ image_names: string[]; total_count: number }>(
    `/api/v1/virtual_boards/by_date/${encodeURIComponent(getDateFromBoardId(boardId))}/image_names?${query}`,
    { signal }
  );

  return {
    imageNames: body.image_names,
    total: normalizeTotal(body.total_count, body.image_names.length),
  };
};

export const hydrateGalleryDateBoardImagePage = async ({
  imageNames,
  limit,
  offset,
  signal,
  total,
}: GalleryDateBoardImageNames & {
  limit: number;
  offset: number;
  signal?: AbortSignal;
}): Promise<GalleryImagesPage> => ({
  images: await getGalleryImagesByNames(imageNames.slice(offset, offset + limit), signal),
  total,
});

export const listGalleryImages = async ({
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
}: {
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
}): Promise<GalleryImagesPage> => {
  if (isDateBoardId(boardId)) {
    const names = await listGalleryDateBoardImageNames({
      boardId,
      createdFrom,
      createdTo,
      galleryView,
      orderDir,
      searchTerm,
      signal,
      starredFirst,
    });

    return hydrateGalleryDateBoardImagePage({ ...names, limit, offset, signal });
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
): Promise<void> => {
  const query = toSearchParams({ include_images: includeImages });

  await apiFetch(`/api/v1/boards/${encodeURIComponent(boardId)}?${query}`, { method: 'DELETE', signal });
};

export const addImagesToGalleryBoard = async (
  boardId: string,
  imageNames: string[],
  signal?: AbortSignal
): Promise<void> => {
  if (
    boardId === 'none' ||
    boardId === 'generated' ||
    boardId === 'assets' ||
    isDateBoardId(boardId) ||
    imageNames.length === 0
  ) {
    return;
  }

  await apiFetchJson('/api/v1/board_images/batch', {
    body: JSON.stringify({ board_id: boardId, image_names: imageNames }),
    method: 'POST',
    signal,
  });
};

export const removeImagesFromGalleryBoard = async (imageNames: string[], signal?: AbortSignal): Promise<void> => {
  if (imageNames.length === 0) {
    return;
  }

  await apiFetchJson('/api/v1/board_images/batch/delete', {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });
};

const setGalleryImagesStarred = async (imageNames: string[], starred: boolean, signal?: AbortSignal): Promise<void> => {
  if (imageNames.length === 0) {
    return;
  }

  await apiFetchJson(`/api/v1/images/${starred ? 'star' : 'unstar'}`, {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });
};

export const starGalleryImages = (imageNames: string[], signal?: AbortSignal): Promise<void> =>
  setGalleryImagesStarred(imageNames, true, signal);

export const unstarGalleryImages = (imageNames: string[], signal?: AbortSignal): Promise<void> =>
  setGalleryImagesStarred(imageNames, false, signal);

interface DeleteImagesResponse {
  deleted_images: string[];
  /** Names the backend could not delete. Absent on backends predating partial-failure reporting. */
  failed_images?: string[];
  affected_boards: string[];
}

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
  if (imageNames.length === 0) {
    return { deletedImageNames: [], failedImageNames: [] };
  }

  const body = await apiFetchJson<DeleteImagesResponse>('/api/v1/images/delete', {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
    signal,
  });

  return {
    deletedImageNames: body.deleted_images,
    failedImageNames: body.failed_images ?? [],
  };
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
    board_id: boardId === 'none' || isDateBoardId(boardId) ? undefined : boardId,
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

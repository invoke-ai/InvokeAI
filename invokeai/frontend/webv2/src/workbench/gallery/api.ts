import { absolutizeApiUrl, apiFetch, apiFetchJson, apiFetchRaw, sleep } from '@workbench/backend/http';
import type { GeneratedImageContract } from '@workbench/types';

export type GalleryView = 'images' | 'assets';

export type GalleryOrderDir = 'ASC' | 'DESC';

export type GalleryBoardOrderBy = 'created_at' | 'board_name';

export interface BackendBoardDTO {
  board_id: string;
  board_name: string;
  image_count: number;
  asset_count: number;
  archived: boolean;
  cover_image_name?: string | null;
  /** Board owner's display name; populated only for admins on multi-user backends. */
  owner_username?: string | null;
}

/**
 * 'board' is a real backend board; 'uncategorized' is the pseudo-board for
 * unassigned images (board_id 'none'); 'date' is a read-only virtual board
 * grouping images by creation date (id 'by_date:YYYY-MM-DD').
 */
export type GalleryBoardKind = 'board' | 'uncategorized' | 'date';

export interface GalleryBoard {
  id: string;
  name: string;
  kind: GalleryBoardKind;
  imageCount: number;
  assetCount: number;
  archived: boolean;
  coverImageName?: string | null;
  coverThumbnailUrl?: string;
  /** Owner display name when the backend exposes it (admins in multi-user mode). */
  ownerName?: string | null;
}

const DATE_BOARD_ID_PREFIX = 'by_date:';

export const isDateBoardId = (boardId: string): boolean => boardId.startsWith(DATE_BOARD_ID_PREFIX);

const getDateFromBoardId = (boardId: string): string => boardId.slice(DATE_BOARD_ID_PREFIX.length);

export interface BackendImageDTO {
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

export interface GalleryImage extends GeneratedImageContract {
  boardId: string;
  imageCategory: BackendImageDTO['image_category'];
  starred: boolean;
}

export type GalleryImageMetadata = Record<string, unknown>;

interface ListImagesResponse {
  items: BackendImageDTO[];
  limit: number;
  offset: number;
  total: number;
}

const imageCategories = ['general'];
const assetCategories = ['control', 'mask', 'user', 'other'];

const getImageThumbnailUrl = (imageName: string): string =>
  absolutizeApiUrl(`/api/v1/images/i/${encodeURIComponent(imageName)}/thumbnail`);

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

const mapBoard = (board: BackendBoardDTO): GalleryBoard => ({
  archived: board.archived,
  assetCount: board.asset_count,
  coverImageName: board.cover_image_name,
  coverThumbnailUrl: board.cover_image_name ? getImageThumbnailUrl(board.cover_image_name) : undefined,
  id: board.board_id,
  imageCount: board.image_count,
  kind: 'board',
  name: board.board_name,
  ownerName: board.owner_username ?? null,
});

const getGalleryTotal = async ({ boardId, categories }: { boardId: string; categories: string[] }): Promise<number> => {
  const query = toSearchParams({
    board_id: boardId,
    categories,
    is_intermediate: false,
    limit: 0,
    offset: 0,
  });
  const body = await apiFetchJson<Pick<ListImagesResponse, 'total'>>(`/api/v1/images/?${query}`);

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

export const listGalleryBoards = async ({
  includeArchived = false,
  orderBy = 'created_at',
  orderDir = 'DESC',
}: {
  includeArchived?: boolean;
  orderBy?: GalleryBoardOrderBy;
  orderDir?: GalleryOrderDir;
} = {}): Promise<GalleryBoard[]> => {
  const boardsQuery = toSearchParams({
    all: true,
    direction: orderDir,
    include_archived: includeArchived,
    order_by: orderBy,
  });
  const boardsBodyPromise = apiFetchJson<BackendBoardDTO[] | { items?: BackendBoardDTO[] }>(
    `/api/v1/boards/?${boardsQuery}`
  );

  const [body, uncategorizedImageCount, uncategorizedAssetCount] = await Promise.all([
    boardsBodyPromise,
    getGalleryTotal({ boardId: 'none', categories: imageCategories }),
    getGalleryTotal({ boardId: 'none', categories: assetCategories }),
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
    },
    ...boards.filter((board) => includeArchived || !board.archived).map(mapBoard),
  ];
};

interface VirtualDateBoardDTO {
  virtual_board_id: string;
  board_name: string;
  date: string;
  image_count: number;
  asset_count: number;
  cover_image_name?: string | null;
}

export const listGalleryDateBoards = async (): Promise<GalleryBoard[]> => {
  const body = await apiFetchJson<VirtualDateBoardDTO[]>('/api/v1/virtual_boards/by_date');

  return body.map((board) => ({
    archived: false,
    assetCount: board.asset_count,
    coverImageName: board.cover_image_name,
    coverThumbnailUrl: board.cover_image_name ? getImageThumbnailUrl(board.cover_image_name) : undefined,
    id: board.virtual_board_id,
    imageCount: board.image_count,
    kind: 'date',
    name: board.board_name,
  }));
};

export interface GalleryImagesPage {
  images: GalleryImage[];
  total: number;
}

const getGalleryImagesByNames = async (imageNames: string[]): Promise<GalleryImage[]> => {
  if (imageNames.length === 0) {
    return [];
  }

  const body = await apiFetchJson<BackendImageDTO[]>('/api/v1/images/images_by_names', {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
  });

  return body.map(mapImage);
};

/**
 * Date virtual boards have no offset-paginated DTO endpoint; list the ordered
 * names for the date, slice the requested window, then bulk-hydrate DTOs.
 */
const listGalleryDateBoardImages = async ({
  boardId,
  galleryView,
  limit,
  offset,
  orderDir,
  searchTerm,
  starredFirst,
}: {
  boardId: string;
  galleryView: GalleryView;
  limit: number;
  offset: number;
  orderDir: GalleryOrderDir;
  searchTerm: string;
  starredFirst: boolean;
}): Promise<GalleryImagesPage> => {
  const query = toSearchParams({
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred_first: starredFirst,
  });
  const body = await apiFetchJson<{ image_names: string[]; total_count: number }>(
    `/api/v1/virtual_boards/by_date/${encodeURIComponent(getDateFromBoardId(boardId))}/image_names?${query}`
  );
  const images = await getGalleryImagesByNames(body.image_names.slice(offset, offset + limit));

  return { images, total: body.total_count };
};

export const listGalleryImages = async ({
  boardId,
  galleryView,
  limit = 100,
  offset = 0,
  orderDir = 'DESC',
  searchTerm,
  starredFirst = false,
}: {
  boardId: string;
  galleryView: GalleryView;
  limit?: number;
  offset?: number;
  orderDir?: GalleryOrderDir;
  searchTerm: string;
  starredFirst?: boolean;
}): Promise<GalleryImagesPage> => {
  if (isDateBoardId(boardId)) {
    return listGalleryDateBoardImages({ boardId, galleryView, limit, offset, orderDir, searchTerm, starredFirst });
  }

  const query = toSearchParams({
    board_id: boardId,
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    is_intermediate: false,
    limit,
    offset,
    order_dir: orderDir,
    search_term: searchTerm.trim() || undefined,
    starred_first: starredFirst,
  });
  const body = await apiFetchJson<ListImagesResponse>(`/api/v1/images/?${query}`);

  return { images: body.items.map(mapImage), total: body.total };
};

export const getGalleryImageMetadata = async (imageName: string): Promise<GalleryImageMetadata | null> => {
  const body = await apiFetchJson<unknown>(`/api/v1/images/i/${encodeURIComponent(imageName)}/metadata`);

  return body && typeof body === 'object' && !Array.isArray(body) ? (body as GalleryImageMetadata) : null;
};

export const createGalleryBoard = async (boardName: string): Promise<GalleryBoard> => {
  const query = toSearchParams({ board_name: boardName });
  const body = await apiFetchJson<BackendBoardDTO>(`/api/v1/boards/?${query}`, { method: 'POST' });

  return mapBoard(body);
};

export const updateGalleryBoard = async (
  boardId: string,
  changes: { name?: string; archived?: boolean }
): Promise<GalleryBoard> => {
  const body = await apiFetchJson<BackendBoardDTO>(`/api/v1/boards/${encodeURIComponent(boardId)}`, {
    body: JSON.stringify({ archived: changes.archived, board_name: changes.name }),
    method: 'PATCH',
  });

  return mapBoard(body);
};

export const deleteGalleryBoard = async (boardId: string, includeImages: boolean): Promise<void> => {
  const query = toSearchParams({ include_images: includeImages });

  await apiFetch(`/api/v1/boards/${encodeURIComponent(boardId)}?${query}`, { method: 'DELETE' });
};

export const addImagesToGalleryBoard = async (boardId: string, imageNames: string[]): Promise<void> => {
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
  });
};

export const removeImagesFromGalleryBoard = async (imageNames: string[]): Promise<void> => {
  if (imageNames.length === 0) {
    return;
  }

  await apiFetchJson('/api/v1/board_images/batch/delete', {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
  });
};

const setGalleryImagesStarred = async (imageNames: string[], starred: boolean): Promise<void> => {
  if (imageNames.length === 0) {
    return;
  }

  await apiFetchJson(`/api/v1/images/${starred ? 'star' : 'unstar'}`, {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
  });
};

export const starGalleryImages = (imageNames: string[]): Promise<void> => setGalleryImagesStarred(imageNames, true);

export const unstarGalleryImages = (imageNames: string[]): Promise<void> => setGalleryImagesStarred(imageNames, false);

export const deleteGalleryImages = async (imageNames: string[]): Promise<void> => {
  if (imageNames.length === 0) {
    return;
  }

  await apiFetchJson('/api/v1/images/delete', {
    body: JSON.stringify({ image_names: imageNames }),
    method: 'POST',
  });
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
}: {
  boardId?: string;
  imageNames?: string[];
}): Promise<{ blob: Blob; fileName: string }> => {
  const { bulk_download_item_name: fileName } = await apiFetchJson<{ bulk_download_item_name?: string | null }>(
    '/api/v1/images/download',
    {
      body: JSON.stringify({ board_id: boardId, image_names: imageNames }),
      method: 'POST',
    }
  );

  if (!fileName) {
    throw new Error('The bulk download failed to start.');
  }

  const startedAt = Date.now();

  while (Date.now() - startedAt < BULK_DOWNLOAD_TIMEOUT_MS) {
    const response = await apiFetchRaw(`/api/v1/images/download/${encodeURIComponent(fileName)}`);

    if (response.ok) {
      return { blob: await response.blob(), fileName };
    }

    if (response.status !== 404) {
      throw new Error(`${response.status} ${response.statusText}`);
    }

    await sleep(BULK_DOWNLOAD_POLL_INTERVAL_MS);
  }

  throw new Error('Timed out preparing the download archive.');
};

export const uploadGalleryImage = async (file: File, boardId: string): Promise<GalleryImage> => {
  const query = toSearchParams({
    board_id: boardId === 'none' || isDateBoardId(boardId) ? undefined : boardId,
    image_category: 'user',
    is_intermediate: false,
  });
  const body = new FormData();
  body.append('file', file);

  const uploadedImage = await apiFetchJson<BackendImageDTO>(`/api/v1/images/upload?${query}`, {
    body,
    method: 'POST',
  });

  return mapImage(uploadedImage);
};

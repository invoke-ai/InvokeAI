import type { GeneratedImageContract } from '../types';

const API_BASE_URL = import.meta.env.VITE_INVOKEAI_API_BASE_URL ?? '';

export type GalleryView = 'images' | 'assets';

export interface BackendBoardDTO {
  board_id: string;
  board_name: string;
  image_count: number;
  asset_count: number;
  archived: boolean;
  cover_image_name?: string | null;
}

export interface GalleryBoard {
  id: string;
  name: string;
  imageCount: number;
  assetCount: number;
  coverImageName?: string | null;
  coverThumbnailUrl?: string;
  isVirtual?: boolean;
}

export interface BackendImageDTO {
  image_name: string;
  image_url: string;
  thumbnail_url: string;
  width: number;
  height: number;
  created_at: string;
  image_category: 'general' | 'control' | 'mask' | 'user' | 'other';
  is_intermediate: boolean;
  board_id?: string | null;
}

export interface GalleryImage extends GeneratedImageContract {
  boardId: string;
  imageCategory: BackendImageDTO['image_category'];
}

interface ListImagesResponse {
  items: BackendImageDTO[];
  limit: number;
  offset: number;
  total: number;
}

const imageCategories = ['general'];
const assetCategories = ['control', 'mask', 'user', 'other'];

const buildUrl = (path: string): string => `${API_BASE_URL}${path}`;

const absolutizeImageUrl = (url: string): string => {
  if (!API_BASE_URL || url.startsWith('http://') || url.startsWith('https://')) {
    return url;
  }

  return new URL(url, API_BASE_URL).toString();
};

const getImageThumbnailUrl = (imageName: string): string =>
  absolutizeImageUrl(`/api/v1/images/i/${encodeURIComponent(imageName)}/thumbnail`);

const assertOk = async (response: Response): Promise<Response> => {
  if (response.ok) {
    return response;
  }

  const text = await response.text();
  throw new Error(text || `${response.status} ${response.statusText}`);
};

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
  assetCount: board.asset_count,
  coverImageName: board.cover_image_name,
  coverThumbnailUrl: board.cover_image_name ? getImageThumbnailUrl(board.cover_image_name) : undefined,
  id: board.board_id,
  imageCount: board.image_count,
  name: board.board_name,
});

const getGalleryTotal = async ({ boardId, categories }: { boardId: string; categories: string[] }): Promise<number> => {
  const query = toSearchParams({
    board_id: boardId,
    categories,
    is_intermediate: false,
    limit: 0,
    offset: 0,
  });
  const response = await assertOk(await fetch(buildUrl(`/api/v1/images/?${query}`)));
  const body = (await response.json()) as Pick<ListImagesResponse, 'total'>;

  return body.total;
};

const mapImage = (image: BackendImageDTO): GalleryImage => ({
  boardId: image.board_id ?? 'none',
  height: image.height,
  imageCategory: image.image_category,
  imageName: image.image_name,
  imageUrl: absolutizeImageUrl(image.image_url),
  queuedAt: image.created_at,
  sourceQueueItemId: 'backend-gallery',
  thumbnailUrl: absolutizeImageUrl(image.thumbnail_url),
  width: image.width,
});

export const listGalleryBoards = async (): Promise<GalleryBoard[]> => {
  const boardsResponsePromise = fetch(buildUrl('/api/v1/boards/?all=true&include_archived=false')).then(assertOk);

  const [response, uncategorizedImageCount, uncategorizedAssetCount] = await Promise.all([
    boardsResponsePromise,
    getGalleryTotal({ boardId: 'none', categories: imageCategories }),
    getGalleryTotal({ boardId: 'none', categories: assetCategories }),
  ]);
  const body = (await response.json()) as BackendBoardDTO[] | { items?: BackendBoardDTO[] };
  const boards = Array.isArray(body) ? body : (body.items ?? []);

  return [
    {
      assetCount: uncategorizedAssetCount,
      id: 'none',
      imageCount: uncategorizedImageCount,
      isVirtual: true,
      name: 'Uncategorized',
    },
    ...boards.filter((board) => !board.archived).map(mapBoard),
  ];
};

export const listGalleryImages = async ({
  boardId,
  galleryView,
  searchTerm,
}: {
  boardId: string;
  galleryView: GalleryView;
  searchTerm: string;
}): Promise<GalleryImage[]> => {
  const query = toSearchParams({
    board_id: boardId,
    categories: galleryView === 'assets' ? assetCategories : imageCategories,
    is_intermediate: false,
    limit: 100,
    offset: 0,
    order_dir: 'DESC',
    search_term: searchTerm.trim() || undefined,
  });
  const response = await assertOk(await fetch(buildUrl(`/api/v1/images/?${query}`)));
  const body = (await response.json()) as ListImagesResponse;

  return body.items.map(mapImage);
};

export const createGalleryBoard = async (boardName: string): Promise<GalleryBoard> => {
  const query = toSearchParams({ board_name: boardName });
  const response = await assertOk(
    await fetch(buildUrl(`/api/v1/boards/?${query}`), {
      method: 'POST',
    })
  );

  return mapBoard((await response.json()) as BackendBoardDTO);
};

export const addImagesToGalleryBoard = async (boardId: string, imageNames: string[]): Promise<void> => {
  if (boardId === 'none' || boardId === 'generated' || boardId === 'assets' || imageNames.length === 0) {
    return;
  }

  await assertOk(
    await fetch(buildUrl('/api/v1/board_images/batch'), {
      body: JSON.stringify({ board_id: boardId, image_names: imageNames }),
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
    })
  );
};

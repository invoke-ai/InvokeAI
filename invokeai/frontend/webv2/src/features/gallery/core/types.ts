export interface GeneratedImageContract {
  height: number;
  imageName: string;
  imageUrl: string;
  queuedAt: string;
  sourceQueueItemId: string;
  thumbnailUrl: string;
  width: number;
}

export type GalleryView = 'images' | 'assets';

export type GalleryOrderDir = 'ASC' | 'DESC';

export type GalleryBoardOrderBy = 'created_at' | 'board_name';

export type GalleryBoardKind = 'board' | 'uncategorized' | 'date';

export interface GalleryBoard {
  id: string;
  name: string;
  kind: GalleryBoardKind;
  imageCount: number;
  assetCount: number;
  /** Videos on the board, counted separately from `imageCount` by the backend. */
  videoCount: number;
  archived: boolean;
  coverImageName?: string | null;
  /** Set instead of `coverImageName` when the board's most recent item is a video. */
  coverVideoName?: string | null;
  coverThumbnailUrl?: string;
  /** ISO creation timestamp; absent for uncategorized and date virtual boards. */
  createdAt?: string | null;
  ownerName?: string | null;
}

export interface GalleryImage extends GeneratedImageContract {
  boardId: string;
  imageCategory: 'general' | 'control' | 'mask' | 'user' | 'other';
  starred: boolean;
}

export type GalleryImageMetadata = Record<string, unknown>;

/**
 * Per-name outcome of a bulk delete. The backend deletes each name independently, so a
 * request can partly succeed and callers must evict only what actually went away.
 */
export interface GalleryDeletionResult {
  deletedImageNames: string[];
  failedImageNames: string[];
}

export interface GalleryImagesPage {
  images: GalleryImage[];
  total: number;
}

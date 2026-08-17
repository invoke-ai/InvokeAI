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
  /**
   * The project that owns this board, or `null` for an ordinary one.
   *
   * A project's board is part of that project: it takes its name from it, is
   * deleted with it, and the generic board routes refuse to rename, archive or
   * delete it. The UI hides those actions for *any* project-owned board, not just
   * the open project's — the server is authoritative either way, so a stale
   * client is refused rather than obeyed.
   */
  projectId: string | null;
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

/**
 * Authoritative outcome of deleting a board. Deleting the board with its
 * contents reports deleted/failed media; retaining its contents reports the
 * removed board relationships instead.
 */
export interface GalleryBoardDeletionResult {
  boardId: string;
  deletedBoardImageNames: string[];
  deletedBoardVideoNames: string[];
  deletedImageNames: string[];
  deletedVideoNames: string[];
  failedImageNames: string[];
  failedVideoNames: string[];
}

export interface GalleryImagesPage {
  images: GalleryImage[];
  total: number;
}

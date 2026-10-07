export interface GeneratedImageContract {
  /** Board the backend saved the image to, when known. */
  boardId?: string;
  /** Backend creation timestamp, when known; `queuedAt` is the (earlier)
   * submission instant, so ordering must prefer this field. */
  createdAt?: string;
  height: number;
  imageName: string;
  imageUrl: string;
  queuedAt: string;
  sourceQueueItemId: string;
  thumbnailUrl: string;
  width: number;
}

/** A video a local run produced, as routed back from the queue. */
export interface GeneratedVideoContract {
  /** Board the backend holds the video on; unset when uncategorized. */
  boardId?: string;
  category: GalleryImage['imageCategory'];
  /** Backend creation timestamp, when known. */
  createdAt?: string;
  durationSeconds: number;
  fps?: number;
  height: number;
  isIntermediate: boolean;
  mediaOrigin?: string;
  queuedAt: string;
  sourceQueueItemId: string;
  thumbnailUrl: string;
  videoName: string;
  videoUrl: string;
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
  /** Asset-category (non-'general') videos on the board; uploaded videos are assets. */
  assetVideoCount: number;
  archived: boolean;
  coverImageName?: string | null;
  /** Set instead of `coverImageName` when the board's most recent item is a video. */
  coverVideoName?: string | null;
  coverThumbnailUrl?: string;
  /** ISO creation timestamp; absent for uncategorized and date virtual boards. */
  createdAt?: string | null;
  ownerName?: string | null;
  /**
   * Project ownership controls naming and deletion; generic board actions must exclude all project-owned boards.
   * Null denotes an ordinary board.
   */
  projectId: string | null;
}

export interface GalleryImage extends GeneratedImageContract {
  boardId: string;
  imageCategory: 'general' | 'control' | 'mask' | 'user' | 'other';
  starred: boolean;
  /** Whether the image embeds the workflow that made it. Image records say; the mixed-media items listing does not, so grid images leave it unset. */
  hasWorkflow?: boolean;
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

/** Options shared by every media deletion. */
export interface GalleryDeleteOptions {
  /** `false` has the backend keep starred media instead of deleting it. Absent means delete everything requested. */
  deleteStarred?: boolean;
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
  /** Starred media kept when protection was on; like the board's other survivors it is now uncategorized. */
  protectedImageNames: string[];
  protectedVideoNames: string[];
}

export interface GalleryImagesPage {
  images: GalleryImage[];
  total: number;
}

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

/** What the gallery needs to know about a project to group its boards: the id boards carry, and a name. */
export interface GalleryProjectRef {
  id: string;
  name: string;
}

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
  /** The owner's user id; a project only ever holds its owner's boards. */
  ownerId?: string;
  ownerName?: string | null;
  /**
   * The project this board belongs to; null is the Library. Members are ordinary boards; only the inbox is
   * managed through its project.
   */
  projectId: string | null;
  /** Its project's inbox: named after the project and only renamed, archived, moved or deleted through it. */
  isInbox: boolean;
  /** Who besides its owner can see it; only a private board can join a project. Absent reads as private. */
  visibility?: 'private' | 'shared' | 'public';
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

import type { ImageCategory } from 'services/api/types';
import z from 'zod';

const zGalleryView = z.enum(['images', 'assets']);
export type GalleryView = z.infer<typeof zGalleryView>;
const zBoardId = z.string();
// TS hack to get autocomplete for "none" but accept any string
export type BoardId = 'none' | (string & {});
const zComparisonMode = z.enum(['slider', 'side-by-side', 'hover']);
export type ComparisonMode = z.infer<typeof zComparisonMode>;
const zComparisonFit = z.enum(['contain', 'fill']);
export type ComparisonFit = z.infer<typeof zComparisonFit>;
const zOrderDir = z.enum(['ASC', 'DESC']);
export type OrderDir = z.infer<typeof zOrderDir>;
const zBoardRecordOrderBy = z.enum(['created_at', 'board_name']);
export type BoardRecordOrderBy = z.infer<typeof zBoardRecordOrderBy>;

export const IMAGE_CATEGORIES: ImageCategory[] = ['general'];
export const ASSETS_CATEGORIES: ImageCategory[] = ['control', 'mask', 'user', 'other'];

export const zGalleryState = z.object({
  selection: z.array(z.string()),
  shouldAutoSwitch: z.boolean(),
  autoAssignBoardOnClick: z.boolean(),
  autoAddBoardId: zBoardId,
  galleryImageMinimumWidth: z.number(),
  selectedBoardId: zBoardId,
  galleryView: zGalleryView,
  boardSearchText: z.string(),
  starredFirst: z.boolean(),
  orderDir: zOrderDir,
  searchTerm: z.string(),
  alwaysShowImageSizeBadge: z.boolean(),
  imageToCompare: z.string().nullable(),
  comparisonMode: zComparisonMode,
  comparisonFit: zComparisonFit,
  shouldShowArchivedBoards: z.boolean(),
  showVirtualBoards: z.boolean(),
  virtualBoardsSectionOpen: z.boolean(),
  boardsListOrderBy: zBoardRecordOrderBy,
  boardsListOrderDir: zOrderDir,
});

export type GalleryState = z.infer<typeof zGalleryState>;

const VIRTUAL_BOARD_ID_PREFIX = 'by_date:';

export const isVirtualBoardId = (id: string): boolean => id.startsWith(VIRTUAL_BOARD_ID_PREFIX);

export const getDateFromVirtualBoardId = (id: string): string => id.replace(VIRTUAL_BOARD_ID_PREFIX, '');

/**
 * The polymorphic gallery treats selection as `string[]` of names. The kind is recoverable from
 * the filename pattern since the backend names images with `.png`, videos with `.mp4` (see
 * SimpleNameService), and canvas projects with a bare UUID (no extension). Centralizing the
 * discriminator here so callers don't have to know about the extension contract.
 */
export const isVideoName = (name: string): boolean => name.toLowerCase().endsWith('.mp4');

// UUID v4 pattern — bare UUIDs are canvas-project names (no extension). Images and videos always
// carry an extension, so checking the absence of a `.` is also a valid discriminator. We use the
// UUID pattern explicitly to defend against future name schemes.
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const isCanvasProjectName = (name: string): boolean => UUID_PATTERN.test(name);

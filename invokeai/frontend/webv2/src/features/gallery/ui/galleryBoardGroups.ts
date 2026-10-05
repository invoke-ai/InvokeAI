import type { GalleryBoard } from '@features/gallery/core/types';

import { getGalleryBoardLabel, type GalleryBoardTranslate } from '@features/gallery/core/boardLabels';

export interface GalleryBoardGroups {
  /** Archived boards, split out of the main list into their own section. */
  archivedBoards: GalleryBoard[];
  /** The search term names no existing board, so it can create one. */
  canCreateFromSearch: boolean;
  dateBoards: GalleryBoard[];
  /** Any row at all survived the search — drives the "no matches" copy. */
  hasAnyMatch: boolean;
  /** Uncategorized first, then the project board (if any), then other boards. */
  yourBoards: GalleryBoard[];
}

/**
 * Apply visibility filters while grouping; GET /boards/ cannot filter other projects and returns the complete
 * list.
 */
export const getGalleryBoardGroups = ({
  boards,
  projectBoardId,
  projectName,
  searchTerm,
  showArchived,
  showDates,
  showOtherProjects,
  t,
}: {
  boards: readonly GalleryBoard[];
  projectBoardId: string | null;
  projectName: string;
  searchTerm: string;
  showArchived: boolean;
  showDates: boolean;
  showOtherProjects: boolean;
  t: GalleryBoardTranslate;
}): GalleryBoardGroups => {
  const normalizedSearchTerm = searchTerm.trim().toLowerCase();
  const matchesSearch = (name: string) => !normalizedSearchTerm || name.toLowerCase().includes(normalizedSearchTerm);

  const uncategorizedBoard = boards.find((board) => board.kind === 'uncategorized') ?? null;
  const fetchedProjectBoard = projectBoardId ? (boards.find((board) => board.id === projectBoardId) ?? null) : null;
  // The board renames with its project server-side, but the fetched list can lag
  // a rename — the live project name is authoritative for the open project's row.
  const projectBoard = fetchedProjectBoard ? { ...fetchedProjectBoard, name: projectName } : null;

  const matchesBoardSearch = (board: GalleryBoard) => matchesSearch(getGalleryBoardLabel(board, t));
  // Always retain the active project's board regardless of the other-projects filter.
  const belongsToVisibleProject = (board: GalleryBoard) =>
    showOtherProjects || board.projectId === null || board.id === projectBoard?.id;
  const regularBoards = boards.filter(
    (board) =>
      board.kind === 'board' &&
      board.id !== projectBoard?.id &&
      belongsToVisibleProject(board) &&
      matchesBoardSearch(board)
  );
  const dateBoards = showDates ? boards.filter((board) => board.kind === 'date' && matchesBoardSearch(board)) : [];
  const archivedBoards = showArchived ? regularBoards.filter((board) => board.archived) : [];

  // Keep the fixed system row first as boards accumulate.
  const yourBoards = [
    ...(uncategorizedBoard && matchesBoardSearch(uncategorizedBoard) ? [uncategorizedBoard] : []),
    ...(projectBoard && matchesBoardSearch(projectBoard) ? [projectBoard] : []),
    ...regularBoards.filter((board) => !board.archived),
  ];

  const hasAnyMatch = yourBoards.length > 0 || dateBoards.length > 0 || archivedBoards.length > 0;
  const hasExactMatch =
    boards.some((board) => getGalleryBoardLabel(board, t).toLowerCase() === normalizedSearchTerm) ||
    projectName.toLowerCase() === normalizedSearchTerm;

  return {
    archivedBoards,
    canCreateFromSearch: normalizedSearchTerm.length > 0 && !hasExactMatch,
    dateBoards,
    hasAnyMatch,
    yourBoards,
  };
};

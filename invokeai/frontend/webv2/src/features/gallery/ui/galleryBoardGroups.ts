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
  /** Project board (if any), then other boards, then Uncategorized last. */
  yourBoards: GalleryBoard[];
}

/**
 * Splits the flat board list into the panel's three sections.
 *
 * Pure and total: it filters on `showArchived`/`showDates`/`showOtherProjects`
 * itself rather than trusting the query to have excluded them, so grouping can
 * be reasoned about (and tested) without a fetch in the picture. The
 * other-projects filter has to happen here in any case — `GET /boards/` takes
 * no project parameter, so the list always arrives complete.
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
  const projectBoard = projectBoardId ? (boards.find((board) => board.id === projectBoardId) ?? null) : null;
  const projectRowName = projectBoard?.name ?? projectName;

  const matchesBoardSearch = (board: GalleryBoard) => matchesSearch(getGalleryBoardLabel(board, t));
  // The open project's own board is never "another project's", whatever the
  // setting says — hiding the board you are working in would be nonsense.
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

  const yourBoards = [
    ...(projectBoard && matchesBoardSearch(projectBoard) ? [projectBoard] : []),
    ...regularBoards.filter((board) => !board.archived),
    ...(uncategorizedBoard && matchesBoardSearch(uncategorizedBoard) ? [uncategorizedBoard] : []),
  ];

  const hasAnyMatch = yourBoards.length > 0 || dateBoards.length > 0 || archivedBoards.length > 0;
  const hasExactMatch =
    boards.some((board) => getGalleryBoardLabel(board, t).toLowerCase() === normalizedSearchTerm) ||
    projectRowName.toLowerCase() === normalizedSearchTerm;

  return {
    archivedBoards,
    canCreateFromSearch: normalizedSearchTerm.length > 0 && !hasExactMatch,
    dateBoards,
    hasAnyMatch,
    yourBoards,
  };
};

import type { GalleryBoard } from '@features/gallery/core/types';

import { getGalleryBoardLabel, inboxFirst, type GalleryBoardTranslate } from '@features/gallery/core/boardLabels';

/** One other project's boards: its inbox first, named after the project, then the rest. */
export interface GalleryBoardProjectGroup {
  boards: GalleryBoard[];
  projectId: string;
}

export interface GalleryBoardGroups {
  /** Archived boards from every tier, split out of the lists into their own section. */
  archivedBoards: GalleryBoard[];
  /** The search term names no existing board, so it can create one. */
  canCreateFromSearch: boolean;
  dateBoards: GalleryBoard[];
  /** Any row at all survived the search — drives the "no matches" copy. */
  hasAnyMatch: boolean;
  /** Uncategorized first, then the boards in no project. */
  libraryBoards: GalleryBoard[];
  /** Other projects in name order; empty unless other projects are shown. */
  otherProjects: GalleryBoardProjectGroup[];
  /** The open project's inbox first, then its other boards. */
  projectBoards: GalleryBoard[];
}

/**
 * Apply visibility filters while grouping; GET /boards/ cannot filter by project and returns the complete list.
 * Membership is the board's own `projectId`; `projectBoardId` also claims the open project's inbox, which a draft
 * project can know before the listing does.
 */
export const getGalleryBoardGroups = ({
  boards,
  projectBoardId,
  projectId,
  searchTerm,
  showArchived,
  showDates,
  showOtherProjects,
  t,
}: {
  boards: readonly GalleryBoard[];
  projectBoardId: string | null;
  /** The open project; null where there is none, in which case every project is "other". */
  projectId: string | null;
  searchTerm: string;
  showArchived: boolean;
  showDates: boolean;
  showOtherProjects: boolean;
  t: GalleryBoardTranslate;
}): GalleryBoardGroups => {
  const normalizedSearchTerm = searchTerm.trim().toLowerCase();
  const matchesBoardSearch = (board: GalleryBoard) =>
    !normalizedSearchTerm || getGalleryBoardLabel(board, t).toLowerCase().includes(normalizedSearchTerm);
  const isOpenProjectBoard = (board: GalleryBoard) =>
    board.id === projectBoardId || (projectId !== null && board.projectId === projectId);

  const uncategorizedBoard = boards.find((board) => board.kind === 'uncategorized') ?? null;
  const regularBoards = boards.filter((board) => board.kind === 'board' && matchesBoardSearch(board));
  const liveBoards = regularBoards.filter((board) => !board.archived);

  const projectBoards = inboxFirst(liveBoards.filter(isOpenProjectBoard));
  const libraryBoards = [
    ...(uncategorizedBoard && matchesBoardSearch(uncategorizedBoard) ? [uncategorizedBoard] : []),
    ...liveBoards.filter((board) => board.projectId === null && !isOpenProjectBoard(board)),
  ];

  const otherProjects: GalleryBoardProjectGroup[] = [];
  if (showOtherProjects) {
    const byProject = new Map<string, GalleryBoard[]>();
    for (const board of liveBoards) {
      if (board.projectId !== null && !isOpenProjectBoard(board)) {
        const group = byProject.get(board.projectId);

        if (group) {
          group.push(board);
        } else {
          byProject.set(board.projectId, [board]);
        }
      }
    }
    for (const [otherProjectId, projectGroupBoards] of byProject) {
      otherProjects.push({ boards: inboxFirst(projectGroupBoards), projectId: otherProjectId });
    }
    // The inbox carries the project's name, so projects sort by it where one is listed.
    const nameOf = (group: GalleryBoardProjectGroup) => group.boards.find((board) => board.isInbox)?.name ?? '';
    otherProjects.sort((left, right) => nameOf(left).localeCompare(nameOf(right)));
  }

  const dateBoards = showDates ? boards.filter((board) => board.kind === 'date' && matchesBoardSearch(board)) : [];
  const archivedBoards = showArchived
    ? regularBoards.filter(
        (board) => board.archived && (showOtherProjects || board.projectId === null || isOpenProjectBoard(board))
      )
    : [];

  const hasAnyMatch =
    projectBoards.length > 0 ||
    libraryBoards.length > 0 ||
    otherProjects.length > 0 ||
    dateBoards.length > 0 ||
    archivedBoards.length > 0;
  // Only the rows on screen count as taken: a name in a hidden project (or an archived board while archived boards
  // are hidden) may legitimately be reused, and refusing it would leave the search with nothing to do.
  const hasExactMatch = [
    ...projectBoards,
    ...libraryBoards,
    ...otherProjects.flatMap((group) => group.boards),
    ...dateBoards,
    ...archivedBoards,
  ].some((board) => getGalleryBoardLabel(board, t).toLowerCase() === normalizedSearchTerm);

  return {
    archivedBoards,
    canCreateFromSearch: normalizedSearchTerm.length > 0 && !hasExactMatch,
    dateBoards,
    hasAnyMatch,
    libraryBoards,
    otherProjects,
    projectBoards,
  };
};

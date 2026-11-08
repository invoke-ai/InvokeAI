import type { GalleryBoard } from '@features/gallery/core/types';

import {
  canOnBoard,
  isOtherUsersGalleryBoard,
  isOwnGalleryBoard,
  isSharedGalleryBoard,
} from '@features/gallery/core/boardAccess';
import {
  getGalleryBoardLabel,
  getGalleryProjectGroupLabel,
  inboxFirst,
  type GalleryBoardTranslate,
} from '@features/gallery/core/boardLabels';

/** One other project's boards: its inbox first, named after the project, then the rest. */
export interface GalleryBoardProjectGroup {
  boards: GalleryBoard[];
  /** The project's name as its heading shows it, which is also the order the groups take. */
  label: string;
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
  /** Other projects in label order; empty unless other projects are shown. */
  otherProjects: GalleryBoardProjectGroup[];
  /** The open project's inbox first, then its other boards. */
  projectBoards: GalleryBoard[];
  /** Other people's boards the caller has a role on, by name; never in the caller's own tiers. */
  sharedBoards: GalleryBoard[];
  /** Other people's boards an admin sees only through admin rights, by name. */
  otherUsersBoards: GalleryBoard[];
}

/**
 * A board renamed, moved, archived and deleted directly, as far as the caller's role allows. An inbox is managed
 * through its project; the open project's is also known by id, because a draft project's inbox can predate the listing.
 */
export const isManagedGalleryBoard = (board: GalleryBoard, projectBoardId: string | null): boolean =>
  board.kind === 'board' && !board.isInbox && board.id !== projectBoardId;

/** A board the caller can move between the Library and their projects: only its owner has projects to move it into. */
export const isMovableGalleryBoard = (board: GalleryBoard, projectBoardId: string | null): boolean =>
  isManagedGalleryBoard(board, projectBoardId) && isOwnGalleryBoard(board) && canOnBoard(board, 'move');

/**
 * Apply visibility filters while grouping; GET /boards/ cannot filter by project and returns the complete list.
 * Membership is the board's own `projectId`; `projectBoardId` also claims the open project's inbox, which a draft
 * project can know before the listing does. Only the caller's own boards join those tiers: someone else's board is
 * listed as shared, or for an admin as another user's, whatever project of theirs it is in.
 */
export const getGalleryBoardGroups = ({
  boards,
  projectBoardId,
  projectId,
  projectNames,
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
  /** The account's project names by id, which name other projects ahead of the names their inboxes were listed with. */
  projectNames: ReadonlyMap<string, string>;
  searchTerm: string;
  showArchived: boolean;
  showDates: boolean;
  showOtherProjects: boolean;
  t: GalleryBoardTranslate;
}): GalleryBoardGroups => {
  const normalizedSearchTerm = searchTerm.trim().toLowerCase();
  // Someone else's board is also found by its owner's name, which its row shows.
  const matchesBoardSearch = (board: GalleryBoard) =>
    !normalizedSearchTerm ||
    getGalleryBoardLabel(board, t).toLowerCase().includes(normalizedSearchTerm) ||
    (!isOwnGalleryBoard(board) && (board.ownerName ?? '').toLowerCase().includes(normalizedSearchTerm));
  const isOpenProjectBoard = (board: GalleryBoard) =>
    board.id === projectBoardId || (projectId !== null && board.projectId === projectId);
  const byName = (left: GalleryBoard, right: GalleryBoard) => left.name.localeCompare(right.name);

  const uncategorizedBoard = boards.find((board) => board.kind === 'uncategorized') ?? null;
  const regularBoards = boards.filter((board) => board.kind === 'board' && matchesBoardSearch(board));
  const liveBoards = regularBoards.filter((board) => !board.archived && isOwnGalleryBoard(board));
  const sharedBoards = regularBoards.filter((board) => !board.archived && isSharedGalleryBoard(board)).sort(byName);
  const otherUsersBoards = regularBoards
    .filter((board) => !board.archived && isOtherUsersGalleryBoard(board))
    .sort(byName);

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
      otherProjects.push({
        boards: inboxFirst(projectGroupBoards),
        // Labelled from every board, not the ones the search kept: a search that hides the inbox must not cost the
        // group the name its inbox would give it.
        label: getGalleryProjectGroupLabel(
          otherProjectId,
          boards.filter((board) => board.projectId === otherProjectId),
          projectNames,
          t
        ),
        projectId: otherProjectId,
      });
    }
    otherProjects.sort((left, right) => left.label.localeCompare(right.label));
  }

  const dateBoards = showDates ? boards.filter((board) => board.kind === 'date' && matchesBoardSearch(board)) : [];
  const archivedBoards = showArchived
    ? regularBoards.filter(
        (board) =>
          board.archived &&
          (!isOwnGalleryBoard(board) || showOtherProjects || board.projectId === null || isOpenProjectBoard(board))
      )
    : [];

  const hasAnyMatch =
    projectBoards.length > 0 ||
    libraryBoards.length > 0 ||
    otherProjects.length > 0 ||
    sharedBoards.length > 0 ||
    otherUsersBoards.length > 0 ||
    dateBoards.length > 0 ||
    archivedBoards.length > 0;
  // Only the caller's own rows on screen count as taken: a name in a hidden project (or an archived board while
  // archived boards are hidden), or someone else's board, may legitimately be reused, and refusing it would leave the
  // search with nothing to do.
  const hasExactMatch = [
    ...projectBoards,
    ...libraryBoards,
    ...otherProjects.flatMap((group) => group.boards),
    ...dateBoards,
    ...archivedBoards.filter(isOwnGalleryBoard),
  ].some((board) => getGalleryBoardLabel(board, t).toLowerCase() === normalizedSearchTerm);

  return {
    archivedBoards,
    canCreateFromSearch: normalizedSearchTerm.length > 0 && !hasExactMatch,
    dateBoards,
    hasAnyMatch,
    libraryBoards,
    otherProjects,
    otherUsersBoards,
    projectBoards,
    sharedBoards,
  };
};

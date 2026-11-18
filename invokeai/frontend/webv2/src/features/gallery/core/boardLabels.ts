import type { GalleryBoard } from './types';

import { canOnBoard, isOtherUsersGalleryBoard, isOwnGalleryBoard, isSharedGalleryBoard } from './boardAccess';
import { ALL_READABLE_BOARDS_ID } from './items';

export type GalleryBoardTranslate = (key: string) => string;

/**
 * Resolves the localized display label for synthesized and stored boards. An inbox is labelled as such: every
 * surface that lists one of the caller's also names its project, as a section, a group or the open project itself.
 * Someone else's inbox, which only an admin sees, keeps its stored name, the project's: nothing else names it.
 */
export const getGalleryBoardLabel = (board: GalleryBoard, t: GalleryBoardTranslate): string =>
  board.kind === 'uncategorized'
    ? t('widgets.gallery.uncategorized')
    : board.isInbox && isOwnGalleryBoard(board)
      ? t('widgets.gallery.inbox')
      : board.name;

/** Inbox first, as the fixed row of its tier. */
export const inboxFirst = (boards: readonly GalleryBoard[]): GalleryBoard[] => [
  ...boards.filter((board) => board.isInbox),
  ...boards.filter((board) => !board.isInbox),
];

/**
 * What a group of another project's boards is headed by. The account's project list is authoritative — it follows a
 * rename at once, where the listing's inbox name lags until boards refetch — then the inbox's stored name, which the
 * server keeps equal to the project's, then a stand-in for a project this account cannot see.
 */
export const getGalleryProjectGroupLabel = (
  projectId: string,
  boards: readonly GalleryBoard[],
  projectNames: ReadonlyMap<string, string>,
  t: GalleryBoardTranslate
): string =>
  projectNames.get(projectId) ?? boards.find((board) => board.isInbox)?.name ?? t('widgets.gallery.unknownProject');

export interface GalleryBoardDestinationGroup {
  boards: GalleryBoard[];
  id: string;
  label: string;
}

/**
 * The tiers a flat "move to board" menu offers, in the panel's order: the open project, the Library, boards shared
 * with the caller, other users' boards (admins), then each other project under its own name. Only boards the caller
 * may add media to are destinations. Empty groups are dropped so a menu never shows a bare heading.
 */
export const getGalleryBoardDestinationGroups = ({
  boards,
  projectId,
  projectName,
  projectNames,
  t,
}: {
  boards: readonly GalleryBoard[];
  projectId: string | null;
  projectName: string;
  projectNames: ReadonlyMap<string, string>;
  t: GalleryBoardTranslate;
}): GalleryBoardDestinationGroup[] => {
  const otherProjects = new Map<string, GalleryBoard[]>();
  const destinations = boards.filter((board) => canOnBoard(board, 'add_media'));
  const own = destinations.filter(isOwnGalleryBoard);
  const byName = (left: GalleryBoard, right: GalleryBoard) => left.name.localeCompare(right.name);

  for (const board of own) {
    if (board.projectId !== null && board.projectId !== projectId) {
      const group = otherProjects.get(board.projectId);

      if (group) {
        group.push(board);
      } else {
        otherProjects.set(board.projectId, [board]);
      }
    }
  }

  const groups: GalleryBoardDestinationGroup[] = [
    {
      boards: inboxFirst(own.filter((board) => projectId !== null && board.projectId === projectId)),
      id: 'project',
      label: projectName,
    },
    {
      boards: own.filter((board) => board.projectId === null),
      id: 'library',
      label: t('widgets.gallery.boardGroups.library'),
    },
    {
      boards: destinations.filter(isSharedGalleryBoard).sort(byName),
      id: 'shared',
      label: t('widgets.gallery.boardGroups.shared'),
    },
    {
      boards: destinations.filter(isOtherUsersGalleryBoard).sort(byName),
      id: 'other-users',
      label: t('widgets.gallery.boardGroups.otherUsers'),
    },
    ...[...otherProjects]
      .map(([otherProjectId, group]) => ({
        boards: inboxFirst(group),
        id: `project:${otherProjectId}`,
        label: getGalleryProjectGroupLabel(otherProjectId, group, projectNames, t),
      }))
      .sort((left, right) => left.label.localeCompare(right.label)),
  ];

  return groups.filter((group) => group.boards.length > 0);
};

/**
 * Where a set of items can be moved, shared by the selection bar and the item menus: stored boards that are not
 * archived (date and Uncategorized entries are not boards), less the one board every item is already on. Removing
 * from a board is offered only while some item may be on one. Items no loaded page holds sit on the listing's board;
 * where that is unknown (no listing named, or a search of every board) they could be on any board.
 */
export const getGalleryMoveTargets = (
  boards: readonly GalleryBoard[],
  {
    itemBoardIds,
    listingBoardId = null,
    unloadedCount = 0,
  }: {
    /** The board of each item a loaded page holds. */
    itemBoardIds: readonly string[];
    /** The board the listing the items were selected in reads. */
    listingBoardId?: string | null;
    /** Selected items no loaded page holds. */
    unloadedCount?: number;
  }
): { boards: GalleryBoard[]; canRemoveFromBoard: boolean } => {
  const sources = new Set(itemBoardIds);
  const isUnloadedBoardKnown = listingBoardId !== null && listingBoardId !== ALL_READABLE_BOARDS_ID;

  if (unloadedCount > 0 && isUnloadedBoardKnown) {
    sources.add(listingBoardId);
  }

  const isAnyBoardUnknown = unloadedCount > 0 && !isUnloadedBoardKnown;
  const sharedSource = !isAnyBoardUnknown && sources.size === 1 ? [...sources][0] : undefined;

  return {
    boards: boards.filter((board) => board.kind === 'board' && !board.archived && board.id !== sharedSource),
    canRemoveFromBoard: isAnyBoardUnknown || [...sources].some((boardId) => boardId !== 'none'),
  };
};

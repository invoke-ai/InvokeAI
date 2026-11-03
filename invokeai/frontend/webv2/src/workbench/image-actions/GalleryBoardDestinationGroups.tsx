import type { GalleryBoard } from '@features/gallery';

import { Menu } from '@chakra-ui/react';
import { getGalleryBoardDestinationGroups, getGalleryBoardLabel } from '@features/gallery/contracts';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useProjectLibrarySelector } from '@workbench/projects/library';
import { useActiveProjectId, useActiveProjectName } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The stored boards a "to board" submenu offers, grouped in the panel's tiers (the open project, the Library,
 * shared, other users', other projects). Only stored boards (`kind: 'board'`, not date or Uncategorized entries) that
 * accept the caller's media (`add_media`) are listed.
 */
export const GalleryBoardDestinationGroups = ({
  boards,
  onSelect,
}: {
  boards: readonly GalleryBoard[];
  onSelect: (board: GalleryBoard) => void;
}) => {
  const { t } = useTranslation();
  const projectId = useActiveProjectId();
  const projectName = useActiveProjectName();
  const librarySummaries = useProjectLibrarySelector((snapshot) => snapshot.summaries);
  const groups = useMemo(
    () =>
      getGalleryBoardDestinationGroups({
        boards: boards.filter((board) => board.kind === 'board'),
        projectId,
        projectName,
        projectNames: new Map(librarySummaries.map((summary) => [summary.id, summary.name])),
        t,
      }),
    [boards, librarySummaries, projectId, projectName, t]
  );

  return groups.map((group) => (
    <Menu.ItemGroup key={group.id}>
      <Menu.ItemGroupLabel>{group.label}</Menu.ItemGroupLabel>
      {group.boards.map((board) => (
        <GalleryBoardDestinationItem key={board.id} board={board} onSelect={onSelect} />
      ))}
    </Menu.ItemGroup>
  ));
};

const GalleryBoardDestinationItem = ({
  board,
  onSelect,
}: {
  board: GalleryBoard;
  onSelect: (board: GalleryBoard) => void;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onSelect(board), [board, onSelect]);

  return (
    <Menu.Item value={`board-${board.id}`} onClick={handleClick}>
      <MiddleTruncate fontSize="md" minW="0" text={getGalleryBoardLabel(board, t)} />
    </Menu.Item>
  );
};

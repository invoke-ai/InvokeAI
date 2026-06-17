import type { GalleryBoard } from '@workbench/gallery/api';

import {
  Badge,
  Flex,
  HStack,
  Icon,
  Image,
  Input,
  InputGroup,
  Menu,
  Portal,
  ScrollArea,
  Stack,
  Text,
} from '@chakra-ui/react';
import { Button, CloseButton, IconButton } from '@workbench/components/ui';
import {
  ArchiveIcon,
  ArrowDownAZIcon,
  ArrowUpAZIcon,
  CalendarIcon,
  ChevronDownIcon,
  ImageIcon,
  MoreVerticalIcon,
  PinIcon,
  PlusIcon,
  type LucideIcon,
  SearchIcon,
  XIcon,
} from 'lucide-react';
import { useRef, useState, type MouseEvent } from 'react';

import { GalleryBoardMenu, type GalleryBoardMenuTarget } from './GalleryBoardMenu';
import { getBoardCounts } from './galleryStateView';
import { useGalleryWidget } from './GalleryWidgetContext';

export const GalleryBoardSelect = () => {
  const { actions, gallery, projectName } = useGalleryWidget();
  const [isOpen, setIsOpen] = useState(false);
  const [boardSearchTerm, setBoardSearchTerm] = useState('');
  const [boardMenuTarget, setBoardMenuTarget] = useState<GalleryBoardMenuTarget | null>(null);
  const boardMenuActiveRef = useRef(false);
  const trimmedSearchTerm = boardSearchTerm.trim();
  const normalizedSearchTerm = trimmedSearchTerm.toLowerCase();
  const matchesSearch = (name: string) => !normalizedSearchTerm || name.toLowerCase().includes(normalizedSearchTerm);

  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId) ?? null;
  const uncategorizedBoard = gallery.boards.find((board) => board.kind === 'uncategorized') ?? null;
  const projectBoard = gallery.projectBoardId
    ? (gallery.boards.find((board) => board.id === gallery.projectBoardId) ?? null)
    : null;
  const projectRowName = projectBoard?.name ?? projectName;
  const dateBoards = gallery.boards.filter((board) => board.kind === 'date' && matchesSearch(board.name));
  const regularBoards = gallery.boards.filter(
    (board) => board.kind === 'board' && board.id !== gallery.projectBoardId && matchesSearch(board.name)
  );
  const showProjectRow = matchesSearch(projectRowName);
  const showUncategorizedRow = uncategorizedBoard !== null && matchesSearch(uncategorizedBoard.name);
  const hasAnyMatch = showProjectRow || showUncategorizedRow || dateBoards.length > 0 || regularBoards.length > 0;
  const hasExactMatch =
    gallery.boards.some((board) => board.name.toLowerCase() === normalizedSearchTerm) ||
    projectRowName.toLowerCase() === normalizedSearchTerm;
  const canCreateFromSearch = trimmedSearchTerm.length > 0 && !hasExactMatch;

  const closeAndReset = () => {
    setIsOpen(false);
    setBoardSearchTerm('');
  };

  const createBoardFromSearch = () => {
    if (!canCreateFromSearch) {
      return;
    }

    closeAndReset();
    void actions.createBoard(trimmedSearchTerm);
  };

  const openBoardMenu = (board: GalleryBoard, x: number, y: number) => {
    setBoardMenuTarget({ board, x, y });
  };

  const clearSearch = () => {
    setBoardSearchTerm('');
  };

  const clearSearchButton = boardSearchTerm ? (
    <CloseButton aria-label="Clear search" size="2xs" onClick={clearSearch} me="-2">
      <XIcon />
    </CloseButton>
  ) : null;

  return (
    <>
      <Menu.Root
        open={isOpen}
        positioning={{ placement: 'bottom-start' }}
        onOpenChange={(event) => {
          if (event.open) {
            setIsOpen(true);
            return;
          }

          // Interactions with the board actions menu (or its dialogs) land
          // outside this dropdown; swallow the resulting close request so the
          // user returns to the still-open board list.
          if (boardMenuActiveRef.current) {
            return;
          }

          closeAndReset();
        }}
      >
        <Menu.Trigger asChild>
          <Button minW="0" size="sm" variant="outline" w="full" px="1">
            {selectedBoard ? (
              <BoardOptionContent
                badge={selectedBoard.id === gallery.projectBoardId ? 'Project' : undefined}
                board={selectedBoard}
                isSelected={false}
              />
            ) : (
              <Text fontSize="xs" me="auto">
                Loading board...
              </Text>
            )}
            <Icon as={ChevronDownIcon} boxSize="3" flexShrink={0} />
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <Menu.Content
              maxH="min(30rem, calc(100vh - 6rem))"
              minW="min(22rem, calc(100vw - 2rem))"
              overflow="hidden"
              p="0"
            >
              <Stack gap="2" p="2">
                <InputGroup startElement={<Icon as={SearchIcon} size="xs" />} endElement={clearSearchButton}>
                  <Input
                    aria-label="Search or create boards"
                    placeholder="Search or create boards"
                    size="sm"
                    value={boardSearchTerm}
                    onChange={(event) => setBoardSearchTerm(event.currentTarget.value)}
                    onKeyDown={(event) => {
                      event.stopPropagation();

                      if (event.key === 'Enter' && !hasAnyMatch) {
                        event.preventDefault();
                        createBoardFromSearch();
                      }
                    }}
                  />
                </InputGroup>
                <BoardListControls />
              </Stack>
              <Menu.Separator borderColor="border.subtle" m="0" />
              <ScrollArea.Root maxH="20rem" size="xs" variant="hover" w="full">
                <ScrollArea.Viewport maxH="inherit" w="full">
                  <ScrollArea.Content py="1">
                    {showProjectRow &&
                      (projectBoard ? (
                        <BoardRow
                          badge="Project"
                          board={projectBoard}
                          isSelected={projectBoard.id === gallery.selectedBoardId}
                          onOpenMenu={(x, y) => openBoardMenu(projectBoard, x, y)}
                          onSelect={() => {
                            closeAndReset();
                            actions.selectBoard(projectBoard.id);
                          }}
                        />
                      ) : (
                        <Menu.Item
                          value="__project-board__"
                          onClick={() => {
                            closeAndReset();
                            void actions.selectProjectBoard();
                          }}
                        >
                          <HStack gap="2" minW="0" w="full">
                            <BoardCoverIcon icon={PinIcon} />
                            <Text flex="1" fontSize="xs" fontWeight="500" minW="0" truncate>
                              {projectRowName}
                            </Text>
                            <Badge colorPalette="blue" flexShrink={0} size="xs" variant="subtle">
                              Project
                            </Badge>
                          </HStack>
                        </Menu.Item>
                      ))}
                    {showUncategorizedRow && uncategorizedBoard && (
                      <BoardRow
                        board={uncategorizedBoard}
                        isSelected={uncategorizedBoard.id === gallery.selectedBoardId}
                        onOpenMenu={(x, y) => openBoardMenu(uncategorizedBoard, x, y)}
                        onSelect={() => {
                          closeAndReset();
                          actions.selectBoard(uncategorizedBoard.id);
                        }}
                      />
                    )}
                    {dateBoards.length > 0 && (
                      <>
                        <BoardGroupLabel>By Date</BoardGroupLabel>
                        {dateBoards.map((board) => (
                          <BoardRow
                            key={board.id}
                            board={board}
                            isSelected={board.id === gallery.selectedBoardId}
                            onSelect={() => {
                              closeAndReset();
                              actions.selectBoard(board.id);
                            }}
                          />
                        ))}
                      </>
                    )}
                    {(regularBoards.length > 0 || canCreateFromSearch) && <BoardGroupLabel>Boards</BoardGroupLabel>}
                    {regularBoards.map((board) => (
                      <BoardRow
                        key={board.id}
                        badge={board.archived ? 'Archived' : undefined}
                        board={board}
                        isSelected={board.id === gallery.selectedBoardId}
                        onOpenMenu={(x, y) => openBoardMenu(board, x, y)}
                        onSelect={() => {
                          closeAndReset();
                          actions.selectBoard(board.id);
                        }}
                      />
                    ))}
                    {!hasAnyMatch && !canCreateFromSearch && (
                      <Text color="fg.subtle" fontSize="2xs" px="3" py="2">
                        No boards match this search.
                      </Text>
                    )}
                    {canCreateFromSearch && (
                      <Menu.Item value="__create-board__" onClick={createBoardFromSearch}>
                        <HStack gap="2" minW="0" w="full">
                          <BoardCoverIcon icon={PlusIcon} />
                          <Text flex="1" fontSize="xs" fontWeight="600" minW="0" truncate>
                            Create board &ldquo;{trimmedSearchTerm}&rdquo;
                          </Text>
                        </HStack>
                      </Menu.Item>
                    )}
                  </ScrollArea.Content>
                </ScrollArea.Viewport>
                <ScrollArea.Scrollbar>
                  <ScrollArea.Thumb />
                </ScrollArea.Scrollbar>
              </ScrollArea.Root>
            </Menu.Content>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <GalleryBoardMenu
        target={boardMenuTarget}
        onActiveChange={(isActive) => {
          boardMenuActiveRef.current = isActive;
        }}
        onClose={() => setBoardMenuTarget(null)}
      />
    </>
  );
};

const BoardListControls = () => {
  const { actions, gallery } = useGalleryWidget();
  const { boardOrderBy, boardOrderDir, showArchivedBoards, showDateBoards } = gallery.settings;
  const isAscending = boardOrderDir === 'ASC';

  return (
    <HStack gap="1" justify="space-between">
      <HStack gap="1">
        <Button
          size="2xs"
          variant={boardOrderBy === 'board_name' ? 'solid' : 'outline'}
          onClick={() => actions.updateSettings({ boardOrderBy: 'board_name' })}
        >
          Name
        </Button>
        <Button
          size="2xs"
          variant={boardOrderBy === 'created_at' ? 'solid' : 'outline'}
          onClick={() => actions.updateSettings({ boardOrderBy: 'created_at' })}
        >
          Created
        </Button>
        <IconButton
          aria-label={isAscending ? 'Sort boards descending' : 'Sort boards ascending'}
          size="2xs"
          variant="outline"
          onClick={() => actions.updateSettings({ boardOrderDir: isAscending ? 'DESC' : 'ASC' })}
        >
          {isAscending ? <ArrowUpAZIcon /> : <ArrowDownAZIcon />}
        </IconButton>
      </HStack>
      <HStack gap="1">
        <IconButton
          aria-label={showDateBoards ? 'Hide date boards' : 'Show date boards'}
          aria-pressed={showDateBoards}
          size="2xs"
          title={showDateBoards ? 'Hide date boards' : 'Show date boards'}
          variant={showDateBoards ? 'solid' : 'outline'}
          onClick={() => actions.updateSettings({ showDateBoards: !showDateBoards })}
        >
          <CalendarIcon />
        </IconButton>
        <IconButton
          aria-label={showArchivedBoards ? 'Hide archived boards' : 'Show archived boards'}
          aria-pressed={showArchivedBoards}
          size="2xs"
          title={showArchivedBoards ? 'Hide archived boards' : 'Show archived boards'}
          variant={showArchivedBoards ? 'solid' : 'outline'}
          onClick={() => actions.updateSettings({ showArchivedBoards: !showArchivedBoards })}
        >
          <ArchiveIcon />
        </IconButton>
      </HStack>
    </HStack>
  );
};

const BoardGroupLabel = ({ children }: { children: string }) => (
  <Text color="fg.subtle" fontSize="2xs" fontWeight="700" px="3" pb="0.5" pt="2" textTransform="uppercase">
    {children}
  </Text>
);

const BoardRow = ({
  badge,
  board,
  isSelected,
  onOpenMenu,
  onSelect,
}: {
  badge?: 'Project' | 'Archived';
  board: GalleryBoard;
  isSelected: boolean;
  onOpenMenu?: (x: number, y: number) => void;
  onSelect: () => void;
}) => (
  <Menu.Item
    css={onOpenMenu ? { '&:hover .board-row-actions': { opacity: 1 } } : undefined}
    value={board.id}
    onClick={onSelect}
    onContextMenu={
      onOpenMenu
        ? (event: MouseEvent) => {
            event.preventDefault();
            event.stopPropagation();
            onOpenMenu(event.clientX, event.clientY);
          }
        : undefined
    }
  >
    <BoardOptionContent badge={badge} board={board} isSelected={isSelected} showOwner />
    {onOpenMenu && (
      <IconButton
        aria-label={`Board actions for ${board.name}`}
        className="board-row-actions"
        flexShrink={0}
        opacity={0}
        size="2xs"
        transition="opacity 0.15s ease"
        variant="ghost"
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();

          const rect = event.currentTarget.getBoundingClientRect();

          onOpenMenu(rect.left, rect.bottom);
        }}
        onPointerDown={(event) => event.stopPropagation()}
        onPointerUp={(event) => event.stopPropagation()}
      >
        <MoreVerticalIcon />
      </IconButton>
    )}
  </Menu.Item>
);

const BoardOptionContent = ({
  badge,
  board,
  isSelected,
  showOwner = false,
}: {
  badge?: 'Project' | 'Archived';
  board: GalleryBoard;
  isSelected: boolean;
  /** Render the owner line (admins on multi-user backends); off in the compact trigger. */
  showOwner?: boolean;
}) => {
  const counts = getBoardCounts(board);

  return (
    <HStack gap="2" minW="0" w="full">
      <BoardCover board={board} />
      <Stack gap="1" minW="0">
        <Text fontSize="xs" fontWeight={isSelected ? '700' : '500'} minW="0" lineHeight={1} truncate>
          {board.name}
        </Text>
        {showOwner && board.ownerName ? (
          <Text color="fg.subtle" fontSize="2xs" minW="0" lineHeight={1} truncate>
            {board.ownerName}
          </Text>
        ) : null}
      </Stack>
      <Flex ms="auto">
        {badge && (
          <Badge colorPalette={badge === 'Project' ? 'blue' : 'gray'} flexShrink={0} size="xs" variant="subtle">
            {badge}
          </Badge>
        )}
        <Badge flexShrink={0} size="xs" variant={isSelected ? 'solid' : 'subtle'}>
          {counts.imageCount} | {counts.assetCount}
        </Badge>
      </Flex>
    </HStack>
  );
};

const BoardCoverIcon = ({ icon }: { icon: LucideIcon }) => (
  <Flex
    align="center"
    bg="bg.emphasized"
    borderWidth="1px"
    borderColor="border.subtle"
    boxSize="7"
    color="fg.subtle"
    flexShrink={0}
    justify="center"
    rounded="md"
  >
    <Icon as={icon} boxSize="4" />
  </Flex>
);

const BoardCover = ({ board }: { board: GalleryBoard }) => {
  if (board.coverThumbnailUrl) {
    return (
      <Image
        alt=""
        bg="bg.emphasized"
        borderWidth="1px"
        borderColor="border.subtle"
        boxSize="7"
        flexShrink={0}
        objectFit="cover"
        rounded="md"
        src={board.coverThumbnailUrl}
      />
    );
  }

  return <BoardCoverIcon icon={board.kind === 'date' ? CalendarIcon : ImageIcon} />;
};

import type { GalleryItemKey } from '@features/gallery/core/items';
import type { GalleryBoardSectionId } from '@features/gallery/core/settings';
import type { GalleryBoard } from '@features/gallery/core/types';

import { HStack, Icon, ScrollArea, Stack, Text } from '@chakra-ui/react';
import { getGalleryProjectGroupLabel } from '@features/gallery/core/boardLabels';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { usePreservedScrollOffset } from '@platform/react/usePreservedScrollOffset';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { PlusIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryBoardProjectGroup } from './galleryBoardGroups';

import { BoardCoverIcon } from './GalleryBoardCover';
import { GalleryBoardFilters } from './GalleryBoardFilters';
import { getGalleryBoardGroups } from './galleryBoardGroups';
import { GalleryBoardMenu, type GalleryBoardMenuTarget } from './GalleryBoardMenu';
import { GalleryBoardRow } from './GalleryBoardRow';
import { GalleryBoardRowShell } from './GalleryBoardRowShell';
import { GalleryBoardSection } from './GalleryBoardSection';
import { focusVisibleOperable, GalleryLoadNotice } from './GalleryLoadError';
import { useGalleryWidget } from './GalleryWidgetContext';

const SCROLL_CONTENT_PROPS = { py: '1' } as const;
const CREATE_ROW_COVER = <BoardCoverIcon icon={PlusIcon} />;

/** Which tier a board typed into the search field is created in; each tier's "+" chooses it. */
type CreateTier = 'library' | 'project';

export const GalleryBoardsPanel = () => {
  const { t } = useTranslation();
  const { actions, boardsState, gallery, projectId, projectName, projectNames } = useGalleryWidget();
  const [searchTerm, setSearchTerm] = useState('');
  const [createTier, setCreateTier] = useState<CreateTier>('project');
  const [boardMenuTarget, setBoardMenuTarget] = useState<GalleryBoardMenuTarget | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const boardsViewportRef = useRef<HTMLDivElement>(null);

  // The shell keeps the gallery mounted across layout switches, and a scroll
  // container that stops being rendered loses its offset outright.
  usePreservedScrollOffset(boardsViewportRef);

  const { collapsedBoardSections, showArchivedBoards, showDateBoards, showOtherProjectBoards } = gallery.settings;

  const groups = useMemo(
    () =>
      getGalleryBoardGroups({
        boards: gallery.boards,
        projectBoardId: gallery.projectBoardId,
        projectId,
        searchTerm,
        showArchived: showArchivedBoards,
        showDates: showDateBoards,
        showOtherProjects: showOtherProjectBoards,
        t,
      }),
    [
      gallery.boards,
      gallery.projectBoardId,
      projectId,
      searchTerm,
      showArchivedBoards,
      showDateBoards,
      showOtherProjectBoards,
      t,
    ]
  );

  const loadedItemBoardIds = useMemo(
    () => new Map<GalleryItemKey, string>(gallery.items.map((item) => [toGalleryItemKey(item), item.boardId])),
    [gallery.items]
  );

  const trimmedSearchTerm = searchTerm.trim();

  const createBoardFromSearch = useCallback(
    (tier: CreateTier) => {
      if (!groups.canCreateFromSearch) {
        return;
      }

      setSearchTerm('');
      void actions.createBoard(trimmedSearchTerm, tier === 'project' ? projectId : null);
    },
    [actions, groups.canCreateFromSearch, projectId, trimmedSearchTerm]
  );

  // Enter creates only with no matches, avoiding accidental near-duplicate boards.
  const handleSubmitSearch = useCallback(() => {
    if (!groups.hasAnyMatch) {
      createBoardFromSearch(createTier);
    }
  }, [createBoardFromSearch, createTier, groups.hasAnyMatch]);

  // A "+" with a typed, unmatched name creates at once; otherwise it picks the tier and asks for a name.
  const handleAddBoard = useCallback(
    (tier: CreateTier) => {
      if (groups.canCreateFromSearch) {
        createBoardFromSearch(tier);
        return;
      }

      setCreateTier(tier);
      searchInputRef.current?.focus();
    },
    [createBoardFromSearch, groups.canCreateFromSearch]
  );
  const handleAddProjectBoard = useCallback(() => handleAddBoard('project'), [handleAddBoard]);
  const handleAddLibraryBoard = useCallback(() => handleAddBoard('library'), [handleAddBoard]);
  const addProjectBoardAction = useMemo(
    () => <AddBoardButton label={t('widgets.gallery.createBoardInProject')} onClick={handleAddProjectBoard} />,
    [handleAddProjectBoard, t]
  );
  const addLibraryBoardAction = useMemo(
    () => <AddBoardButton label={t('widgets.gallery.createBoardInLibrary')} onClick={handleAddLibraryBoard} />,
    [handleAddLibraryBoard, t]
  );
  const createProjectBoardFromSearch = useCallback(() => createBoardFromSearch('project'), [createBoardFromSearch]);
  const createLibraryBoardFromSearch = useCallback(() => createBoardFromSearch('library'), [createBoardFromSearch]);

  const handleSelectBoard = useCallback(
    (boardId: string) => {
      setSearchTerm('');
      actions.selectBoard(boardId);
    },
    [actions]
  );

  const openBoardMenu = useCallback((board: GalleryBoard, x: number, y: number) => {
    setBoardMenuTarget({ board, x, y });
  }, []);

  const handleBoardMenuClose = useCallback(() => setBoardMenuTarget(null), []);

  const handleToggleSection = useCallback(
    (sectionId: GalleryBoardSectionId, isOpen: boolean) => {
      const nextSections = isOpen
        ? collapsedBoardSections.filter((entry) => entry !== sectionId)
        : [...collapsedBoardSections, sectionId];

      actions.updateSettings({ collapsedBoardSections: nextSections });
    },
    [actions, collapsedBoardSections]
  );

  const isSectionOpen = (sectionId: GalleryBoardSectionId) => !collapsedBoardSections.includes(sectionId);
  // Without the backend's list, the placeholder Uncategorized row (and "no matches") would claim there is nothing
  // else; the failure takes the rows' place instead. While it loads, the fixed rows alone would read as empty tiers.
  const isBoardListUnavailable = boardsState.status === 'error';
  const isBoardListSettled = boardsState.status !== 'loading' && !isBoardListUnavailable;
  const focusBoardList = useCallback(() => focusVisibleOperable(boardsViewportRef.current), []);

  const renderRow = (board: GalleryBoard, accessibleName?: string) => (
    <GalleryBoardRow
      key={board.id}
      accessibleName={accessibleName}
      board={board}
      isAutoAddTarget={board.id === gallery.settings.autoAddBoardId}
      isMenuOpen={boardMenuTarget?.board.id === board.id}
      isSelected={board.id === gallery.selectedBoardId}
      loadedItemBoardIds={loadedItemBoardIds}
      onOpenMenu={openBoardMenu}
      onSelectBoard={handleSelectBoard}
    />
  );
  const renderCreateRow = (tier: CreateTier) =>
    groups.canCreateFromSearch && createTier === tier ? (
      <GalleryBoardRowShell
        cover={CREATE_ROW_COVER}
        label={t('widgets.gallery.createBoardNamedIn', {
          destination: tier === 'project' ? projectName : t('widgets.gallery.boardGroups.library'),
          name: trimmedSearchTerm,
        })}
        labelWeight="600"
        onSelect={tier === 'project' ? createProjectBoardFromSearch : createLibraryBoardFromSearch}
      />
    ) : null;
  // One line, shown while a tier holds only its fixed row, so the two tiers explain themselves once.
  const renderHint = (text: string) =>
    trimmedSearchTerm ? null : (
      <Text color="fg.muted" fontSize="xs" pe="2" ps="2" py="1">
        {text}
      </Text>
    );
  const otherProjectLabel = (group: GalleryBoardProjectGroup) =>
    getGalleryProjectGroupLabel(group.projectId, group.boards, projectNames, t);

  const projectBoards = isBoardListUnavailable ? [] : groups.projectBoards;
  const libraryBoards = isBoardListUnavailable ? [] : groups.libraryBoards;
  // Each tier counts the boards made in it; its fixed row (Inbox, Uncategorized) is not one.
  const countCreated = (boards: GalleryBoard[]) =>
    boards.filter((board) => board.kind === 'board' && !board.isInbox).length;

  return (
    <Stack flex="1" gap="1" minH="0" minW="0">
      <GalleryBoardFilters
        ref={searchInputRef}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        onSubmitSearch={handleSubmitSearch}
      />
      <ScrollArea.Root flex="1" minH="0" variant="hover" w="full">
        <ScrollArea.Viewport ref={boardsViewportRef} h="full" w="full">
          <ScrollArea.Content {...SCROLL_CONTENT_PROPS}>
            <GalleryBoardSection
              action={addProjectBoardAction}
              count={countCreated(projectBoards)}
              isOpen={isSectionOpen('project')}
              label={projectName}
              sectionId="project"
              onToggle={handleToggleSection}
            >
              {boardsState.status === 'error' || boardsState.status === 'stale-error' ? (
                <GalleryLoadNotice
                  message={t(
                    isBoardListUnavailable ? 'widgets.gallery.boardsLoadFailed' : 'widgets.gallery.boardsRefreshFailed'
                  )}
                  pe="1"
                  ps="2"
                  read={boardsState}
                  retryLabel={t('widgets.gallery.retryLoadingBoards')}
                  onFocusLost={focusBoardList}
                />
              ) : null}
              {projectBoards.map((board) => renderRow(board))}
              {renderCreateRow('project')}
              {isBoardListSettled && projectBoards.every((board) => board.isInbox)
                ? renderHint(t('widgets.gallery.projectBoardsHint'))
                : null}
            </GalleryBoardSection>

            <GalleryBoardSection
              action={addLibraryBoardAction}
              count={countCreated(libraryBoards)}
              isOpen={isSectionOpen('library')}
              label={t('widgets.gallery.boardGroups.library')}
              sectionId="library"
              onToggle={handleToggleSection}
            >
              {libraryBoards.map((board) => renderRow(board))}
              {renderCreateRow('library')}
              {isBoardListSettled && libraryBoards.every((board) => board.kind === 'uncategorized')
                ? renderHint(t('widgets.gallery.libraryBoardsHint'))
                : null}
            </GalleryBoardSection>

            {!isBoardListUnavailable && groups.otherProjects.length > 0 ? (
              <GalleryBoardSection
                count={groups.otherProjects.length}
                isOpen={isSectionOpen('other-projects')}
                label={t('widgets.gallery.boardGroups.otherProjects')}
                sectionId="other-projects"
                onToggle={handleToggleSection}
              >
                {groups.otherProjects.map((group) => (
                  <Stack key={group.projectId} gap="0.5">
                    <Text
                      color="fg.muted"
                      fontSize="xs"
                      fontWeight="600"
                      pe="2"
                      ps="2"
                      pt="1"
                      role="heading"
                      aria-level={4}
                      truncate
                    >
                      {otherProjectLabel(group)}
                    </Text>
                    {group.boards.map((board) =>
                      // Every other project's inbox is called "Inbox"; assistive tech needs the project in the name.
                      renderRow(
                        board,
                        board.isInbox ? t('widgets.gallery.inboxOf', { project: otherProjectLabel(group) }) : undefined
                      )
                    )}
                  </Stack>
                ))}
              </GalleryBoardSection>
            ) : null}

            {!isBoardListUnavailable && groups.dateBoards.length > 0 ? (
              <GalleryBoardSection
                isOpen={isSectionOpen('dates')}
                label={t('widgets.gallery.boardGroups.byDate')}
                sectionId="dates"
                onToggle={handleToggleSection}
              >
                {groups.dateBoards.map((board) => (
                  <GalleryBoardRow
                    key={board.id}
                    board={board}
                    isMenuOpen={boardMenuTarget?.board.id === board.id}
                    isSelected={board.id === gallery.selectedBoardId}
                    loadedItemBoardIds={loadedItemBoardIds}
                    onSelectBoard={handleSelectBoard}
                  />
                ))}
              </GalleryBoardSection>
            ) : null}

            {!isBoardListUnavailable && groups.archivedBoards.length > 0 ? (
              <GalleryBoardSection
                isOpen={isSectionOpen('archived')}
                label={t('common.archived')}
                sectionId="archived"
                onToggle={handleToggleSection}
              >
                {groups.archivedBoards.map((board) => renderRow(board))}
              </GalleryBoardSection>
            ) : null}

            {!isBoardListUnavailable && !groups.hasAnyMatch && !groups.canCreateFromSearch ? (
              <HStack justify="center" py="3">
                <Text color="fg.muted" fontSize="xs">
                  {t('widgets.gallery.noBoardsMatchSearch')}
                </Text>
              </HStack>
            ) : null}
          </ScrollArea.Content>
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar>
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>
      <GalleryBoardMenu target={boardMenuTarget} onClose={handleBoardMenuClose} />
    </Stack>
  );
};

const AddBoardButton = ({ label, onClick }: { label: string; onClick: () => void }): ReactNode => (
  <Tooltip content={label}>
    <IconButton aria-label={label} color="fg.muted" size="sm" variant="ghost" onClick={onClick}>
      <Icon as={PlusIcon} boxSize="3.5" />
    </IconButton>
  </Tooltip>
);

import type { GalleryItemKey } from '@features/gallery/core/items';
import type { GalleryBoardSectionId } from '@features/gallery/core/settings';
import type { GalleryBoard } from '@features/gallery/core/types';

import { HStack, Icon, ScrollArea, Stack, Text } from '@chakra-ui/react';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { usePreservedScrollOffset } from '@platform/react/usePreservedScrollOffset';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { PlusIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { BoardCoverIcon } from './GalleryBoardCover';
import { GalleryBoardFilters } from './GalleryBoardFilters';
import { getGalleryBoardGroups } from './galleryBoardGroups';
import { GalleryBoardMenu, type GalleryBoardMenuTarget } from './GalleryBoardMenu';
import { GalleryBoardRow } from './GalleryBoardRow';
import { GalleryBoardRowShell } from './GalleryBoardRowShell';
import { GalleryBoardSection } from './GalleryBoardSection';
import { useGalleryWidget } from './GalleryWidgetContext';

const SCROLL_CONTENT_PROPS = { py: '1' } as const;
const CREATE_ROW_COVER = <BoardCoverIcon icon={PlusIcon} />;

/**
 * The board browser: search and list controls above a grouped, collapsible
 * list of boards. Identical in both layouts — the shells only decide whether it
 * sits above the items area or beside it.
 */
export const GalleryBoardsPanel = () => {
  const { t } = useTranslation();
  const { actions, gallery, projectName } = useGalleryWidget();
  const [searchTerm, setSearchTerm] = useState('');
  const [boardMenuTarget, setBoardMenuTarget] = useState<GalleryBoardMenuTarget | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const boardsViewportRef = useRef<HTMLDivElement>(null);

  // The shell keeps the gallery mounted across layout switches, and a scroll
  // container that stops being rendered loses its offset outright.
  usePreservedScrollOffset(boardsViewportRef);

  const { collapsedBoardSections, showArchivedBoards, showDateBoards } = gallery.settings;

  const groups = useMemo(
    () =>
      getGalleryBoardGroups({
        boards: gallery.boards,
        projectBoardId: gallery.projectBoardId,
        projectName,
        searchTerm,
        showArchived: showArchivedBoards,
        showDates: showDateBoards,
        t,
      }),
    [gallery.boards, gallery.projectBoardId, projectName, searchTerm, showArchivedBoards, showDateBoards, t]
  );

  const loadedItemBoardIds = useMemo(
    () => new Map<GalleryItemKey, string>(gallery.items.map((item) => [toGalleryItemKey(item), item.boardId])),
    [gallery.items]
  );

  const trimmedSearchTerm = searchTerm.trim();

  const createBoardFromSearch = useCallback(() => {
    if (!groups.canCreateFromSearch) {
      return;
    }

    setSearchTerm('');
    void actions.createBoard(trimmedSearchTerm);
  }, [actions, groups.canCreateFromSearch, trimmedSearchTerm]);

  // Enter only creates when nothing matched; with matches on screen it would
  // be far too easy to make a near-duplicate board by reflex.
  const handleSubmitSearch = useCallback(() => {
    if (!groups.hasAnyMatch) {
      createBoardFromSearch();
    }
  }, [createBoardFromSearch, groups.hasAnyMatch]);

  // "+" creates straight away when the field already names the board; with an
  // empty field there is nothing to name it, so it hands over the caret.
  const handleAddBoard = useCallback(() => {
    if (groups.canCreateFromSearch) {
      createBoardFromSearch();
      return;
    }

    searchInputRef.current?.focus();
  }, [createBoardFromSearch, groups.canCreateFromSearch]);

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

  const addBoardAction = useMemo(
    () => (
      <Tooltip content={t('widgets.gallery.createBoard')}>
        <IconButton
          aria-label={t('widgets.gallery.createBoard')}
          color="fg.muted"
          size="2xs"
          variant="ghost"
          onClick={handleAddBoard}
        >
          <Icon as={PlusIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    ),
    [handleAddBoard, t]
  );

  return (
    <Stack flex="1" gap="1" minH="0" minW="0">
      <GalleryBoardFilters
        ref={searchInputRef}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        onSubmitSearch={handleSubmitSearch}
      />
      <ScrollArea.Root flex="1" minH="0" size="xs" variant="hover" w="full">
        <ScrollArea.Viewport ref={boardsViewportRef} h="full" w="full">
          <ScrollArea.Content {...SCROLL_CONTENT_PROPS}>
            <GalleryBoardSection
              action={addBoardAction}
              isOpen={isSectionOpen('boards')}
              label={t('widgets.gallery.boardGroups.boards')}
              sectionId="boards"
              onToggle={handleToggleSection}
            >
              {groups.yourBoards.map((board) => (
                <GalleryBoardRow
                  key={board.id}
                  board={board}
                  isMenuOpen={boardMenuTarget?.board.id === board.id}
                  isSelected={board.id === gallery.selectedBoardId}
                  loadedItemBoardIds={loadedItemBoardIds}
                  onOpenMenu={openBoardMenu}
                  onSelectBoard={handleSelectBoard}
                />
              ))}
              {groups.canCreateFromSearch ? (
                <GalleryBoardRowShell
                  cover={CREATE_ROW_COVER}
                  label={t('widgets.gallery.createBoardNamed', { name: trimmedSearchTerm })}
                  labelWeight="600"
                  onSelect={createBoardFromSearch}
                />
              ) : null}
            </GalleryBoardSection>

            {groups.dateBoards.length > 0 ? (
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

            {groups.archivedBoards.length > 0 ? (
              <GalleryBoardSection
                isOpen={isSectionOpen('archived')}
                label={t('common.archived')}
                sectionId="archived"
                onToggle={handleToggleSection}
              >
                {groups.archivedBoards.map((board) => (
                  <GalleryBoardRow
                    key={board.id}
                    board={board}
                    isMenuOpen={boardMenuTarget?.board.id === board.id}
                    isSelected={board.id === gallery.selectedBoardId}
                    loadedItemBoardIds={loadedItemBoardIds}
                    onOpenMenu={openBoardMenu}
                    onSelectBoard={handleSelectBoard}
                  />
                ))}
              </GalleryBoardSection>
            ) : null}

            {!groups.hasAnyMatch && !groups.canCreateFromSearch ? (
              <HStack justify="center" py="3">
                <Text color="fg.muted" fontSize="2xs">
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

import type { GalleryItem } from '@features/gallery/core/items';
import type { GallerySparseListing } from '@features/gallery/ui/useGalleryData';
import type { KeyboardEvent, RefObject } from 'react';

import { Box, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { getGalleryUploadAccept, toGalleryItemKey } from '@features/gallery/core/items';
import { GALLERY_PAGE_SIZE, galleryItemsPageOptions } from '@features/gallery/data/queries';
import { BoardCover, BoardCoverIcon } from '@features/gallery/ui/GalleryBoardCover';
import { getGalleryBoardGroups } from '@features/gallery/ui/galleryBoardGroups';
import { GalleryAnnouncer, GalleryLoadErrorState, GalleryLoadNotice } from '@features/gallery/ui/GalleryLoadError';
import { GallerySearchField } from '@features/gallery/ui/GallerySearchField';
import { GallerySearchHelp } from '@features/gallery/ui/GallerySearchHelp';
import { getGalleryProjectBoardId, getGallerySelectedBoardId } from '@features/gallery/ui/galleryStateView';
import { useGalleryHost } from '@features/gallery/ui/GalleryUiContext';
import { getGalleryUploadTargetLabel } from '@features/gallery/ui/GalleryUploadButton';
import { GalleryViewSegmentTabs } from '@features/gallery/ui/GalleryViewTabs';
import { useGalleryUploadAction } from '@features/gallery/ui/useGalleryUploadAction';
import { useGalleryUploadInput } from '@features/gallery/ui/useGalleryUploadInput';
import { isImeComposing } from '@platform/browser/imeComposition';
import { Button, CloseButton, IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { Tooltip } from '@platform/ui/Tooltip';
import { ChevronsDownUpIcon, ChevronsUpDownIcon, ExternalLinkIcon, ImageIcon, UploadIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  GALLERY_PICKER_MIN_COLUMNS,
  getGalleryPickerNeighborIndex,
  getGalleryPickerRemaining,
  getGalleryPickerSelectionAfterPick,
  getGalleryPickerStatus,
  getGalleryPickerTileState,
  isGalleryPickerFull,
  isGalleryPickerNavKey,
  isGalleryPickerNavKeyForField,
  type GalleryPickerAccept,
  type GalleryPickerSelection,
} from './galleryPicker';
import { GalleryPickerBoards, type GalleryPickerBoardGroup } from './GalleryPickerBoards';
import { galleryPickerOptionId, GalleryPickerGrid } from './GalleryPickerGrid';
import { useGalleryPickerScope } from './useGalleryPickerScope';

const BOARD_BUTTON_EXPANDED_PROPS = { bg: 'bg.hover' } as const;

const EMPTY_SLOTS: ReadonlyMap<number, GalleryItem> = new Map();
const EMPTY_PAGE_STATES: GallerySparseListing['pageStates'] = new Map();

const haveSameSlots = (left: GallerySparseListing | null, right: GallerySparseListing): boolean => {
  if (left?.total !== right.total || left.itemSlots.size !== right.itemSlots.size) {
    return false;
  }

  for (const [index, item] of right.itemSlots) {
    if (left.itemSlots.get(index) !== item) {
      return false;
    }
  }

  return true;
};

export const GalleryPickerView = ({
  accept,
  searchInputRef,
  selection,
  onClose,
  onPick,
}: {
  accept: GalleryPickerAccept;
  searchInputRef: RefObject<HTMLInputElement | null>;
  selection: GalleryPickerSelection;
  onClose: () => void;
  onPick: (item: GalleryItem) => void;
}) => {
  const { t } = useTranslation();
  const { galleryValues, projectName, revealInGallery } = useGalleryHost();
  const { data, gallerySelectedItem, scope, selectBoard, setSearchTerm, setView, settings, togglePane } =
    useGalleryPickerScope();
  const idBase = useId();
  const listboxId = `${idBase}-options`;
  const boardsId = `${idBase}-boards`;
  const isMultiple = selection.mode === 'multiple';
  const isSearching = scope.searchTerm.trim() !== '';
  const currentKey = gallerySelectedItem ? toGalleryItemKey(gallerySelectedItem) : null;
  const seedKey = gallerySelectedItem && accept.includes(gallerySelectedItem.kind) ? currentKey : null;

  const [activeCursor, setActiveCursor] = useState<{ filterIdentity: string; index: number } | null>(null);
  const [columnCount, setColumnCount] = useState(GALLERY_PICKER_MIN_COLUMNS);
  const [isUploading, setIsUploading] = useState(false);
  // Async uploads need current selection capacity.
  const selectionRef = useRef(selection);
  // Keep prior absolute slots dimmed during scope changes; a new scope starts at its own page zero.
  const [lastListing, setLastListing] = useState<{ filterIdentity: string; listing: GallerySparseListing } | null>(
    null
  );

  // eslint-disable-next-line react/refs
  selectionRef.current = selection;

  const filterIdentity = JSON.stringify(galleryItemsPageOptions(data.filter, 0).queryKey.slice(3, 5));
  if (
    data.items !== null &&
    data.sparseListing &&
    (lastListing?.filterIdentity !== filterIdentity || !haveSameSlots(lastListing.listing, data.sparseListing))
  ) {
    setLastListing({ filterIdentity, listing: data.sparseListing });
  }

  const knownTotal = data.total;
  const isStale =
    data.listing.status === 'loading' &&
    data.items === null &&
    lastListing !== null &&
    lastListing.filterIdentity !== filterIdentity;
  const sparseListing = isStale ? lastListing?.listing : data.sparseListing;
  const itemSlots = sparseListing?.itemSlots ?? EMPTY_SLOTS;
  const pageStates = isStale ? EMPTY_PAGE_STATES : (sparseListing?.pageStates ?? EMPTY_PAGE_STATES);
  const total = sparseListing?.total ?? knownTotal;
  const totalSlots = total ?? GALLERY_PAGE_SIZE;
  const activeFilterCursor = activeCursor?.filterIdentity === filterIdentity ? activeCursor.index : null;
  const setVisibleRange = data.setVisibleRange;

  const defaultActiveIndex = useMemo(() => {
    const slots = [...itemSlots].sort(([left], [right]) => left - right);
    const firstPickable = slots.find(([, item]) => getGalleryPickerTileState(item, accept, selection) === 'pickable');

    return firstPickable?.[0] ?? slots[0]?.[0] ?? -1;
  }, [accept, itemSlots, selection]);

  // Seed once page zero lands. A selection elsewhere in the listing must not pull the picker away from its start.
  if (activeFilterCursor === null && data.items !== null) {
    const seededIndex = seedKey
      ? [...itemSlots].find(([index, item]) => index < GALLERY_PAGE_SIZE && toGalleryItemKey(item) === seedKey)?.[0]
      : undefined;

    setActiveCursor({ filterIdentity, index: seededIndex ?? defaultActiveIndex });
  }

  const resolvedActiveIndex = activeFilterCursor !== null ? activeFilterCursor : defaultActiveIndex;
  const activeItem = resolvedActiveIndex >= 0 ? itemSlots.get(resolvedActiveIndex) : undefined;
  const activePageOffset = Math.floor(Math.max(0, resolvedActiveIndex) / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
  const activePageState = pageStates.get(activePageOffset);
  const selectedBoard = data.boards.find((board) => board.id === data.selectedBoardId);
  const boardName = selectedBoard ? getGalleryBoardLabel(selectedBoard, t) : t('widgets.gallery.selectedBoardFallback');
  const uploadTarget = getGalleryUploadTargetLabel(data.boards, data.selectedBoardId, t);
  const showsGrid = scope.pane === 'items' && data.listing.status !== 'error' && (total === null || total > 0);
  const isBoardListUnavailable = data.boardsState.status === 'error';

  const getTileState = useCallback(
    (item: GalleryItem) => getGalleryPickerTileState(item, accept, selection),
    [accept, selection]
  );

  const pickItem = useCallback(
    (item: GalleryItem) => {
      const current = selectionRef.current;

      if (getGalleryPickerTileState(item, accept, current) !== 'pickable') {
        return;
      }

      onPick(item);

      if (current.mode === 'single' || isGalleryPickerFull(accept, getGalleryPickerSelectionAfterPick(current, item))) {
        onClose();
      }
    },
    [accept, onClose, onPick]
  );

  const uploadFiles = useGalleryUploadAction({ boards: data.boards, selectedBoardId: data.selectedBoardId });
  const adoptUploads = useCallback(
    (uploaded: GalleryItem[]) => {
      let current = selectionRef.current;
      let picked = false;

      for (const item of uploaded) {
        if (getGalleryPickerTileState(item, accept, current) !== 'pickable') {
          continue;
        }

        onPick(item);
        picked = true;
        current = getGalleryPickerSelectionAfterPick(current, item);
      }

      if (picked && (current.mode === 'single' || isGalleryPickerFull(accept, current))) {
        onClose();
      }
    },
    [accept, onClose, onPick]
  );
  const handleUpload = useCallback(
    (files: File[]) => {
      setIsUploading(true);

      return uploadFiles(files)
        .then(adoptUploads)
        .finally(() => setIsUploading(false));
    },
    [adoptUploads, uploadFiles]
  );
  const uploadOptions = useMemo(
    () => ({ accept: getGalleryUploadAccept(accept), multiple: isMultiple }),
    [accept, isMultiple]
  );
  const { inputProps: uploadInputProps, openPicker: openUploadPicker } = useGalleryUploadInput(
    handleUpload,
    uploadOptions
  );

  const boardGroups = useMemo<GalleryPickerBoardGroup[]>(() => {
    const groups = getGalleryBoardGroups({
      boards: data.boards,
      projectBoardId: getGalleryProjectBoardId(galleryValues),
      projectName,
      searchTerm: scope.searchTerm,
      showArchived: settings.showArchivedBoards,
      showDates: settings.showDateBoards,
      // A pick can come from any project's board, whatever the Gallery widget hides for browsing.
      showOtherProjects: true,
      t,
    });

    return [
      { boards: groups.yourBoards, id: 'boards', label: t('widgets.gallery.boardGroups.boards') },
      { boards: groups.dateBoards, id: 'dates', label: t('widgets.gallery.boardGroups.byDate') },
      { boards: groups.archivedBoards, id: 'archived', label: t('common.archived') },
    ].filter((group) => group.boards.length > 0);
  }, [
    data.boards,
    galleryValues,
    projectName,
    scope.searchTerm,
    settings.showArchivedBoards,
    settings.showDateBoards,
    t,
  ]);
  const visibleBoardCount = boardGroups.reduce((count, group) => count + group.boards.length, 0);

  const focusSearch = useCallback(() => searchInputRef.current?.focus(), [searchInputRef]);
  // The view is lazy-loaded, so it arrives after the popover's own initial
  // focus pass; the field takes focus itself as it mounts.
  const attachSearchInput = useCallback(
    (node: HTMLInputElement | null) => {
      searchInputRef.current = node;
      node?.focus();
    },
    [searchInputRef]
  );
  // The clicked row unmounts with the pane; without this, focus falls to the
  // body and the next keystrokes reach the workbench hotkeys.
  const handleSelectBoard = useCallback(
    (boardId: string) => {
      selectBoard(boardId);
      focusSearch();
    },
    [focusSearch, selectBoard]
  );

  const handleSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (isImeComposing(event.nativeEvent)) {
        return;
      }

      if (scope.pane === 'boards') {
        if (event.key === 'ArrowDown') {
          event.preventDefault();
          event.stopPropagation();
          document.getElementById(boardsId)?.querySelector('button')?.focus();
        } else if (event.key === 'Enter' && isSearching) {
          const first = boardGroups[0]?.boards[0];

          event.preventDefault();
          event.stopPropagation();

          if (first) {
            handleSelectBoard(first.id);
          }
        }

        return;
      }

      if (isStale || data.listing.status === 'error') {
        if (event.key === 'Enter' || isGalleryPickerNavKey(event.key)) {
          event.preventDefault();
          event.stopPropagation();
        }
        return;
      }

      if (isGalleryPickerNavKey(event.key) && isGalleryPickerNavKeyForField(event.key, event.currentTarget.value)) {
        const nextIndex = getGalleryPickerNeighborIndex(resolvedActiveIndex, totalSlots, columnCount, event.key);

        event.preventDefault();
        event.stopPropagation();

        if (nextIndex >= 0) {
          setActiveCursor({ filterIdentity, index: nextIndex });
          setVisibleRange?.({ endIndexExclusive: nextIndex + 1, startIndex: nextIndex });
        }
      } else if (event.key === 'Enter') {
        event.preventDefault();
        event.stopPropagation();

        if (activeItem) {
          pickItem(activeItem);
        } else if (activePageState?.error) {
          void activePageState.retry();
        }
      }
    },
    [
      activePageState,
      activeItem,
      boardGroups,
      boardsId,
      columnCount,
      handleSelectBoard,
      isSearching,
      isStale,
      filterIdentity,
      data.listing.status,
      pickItem,
      resolvedActiveIndex,
      setVisibleRange,
      scope.pane,
      totalSlots,
    ]
  );

  const handleActivate = useCallback(
    (item: GalleryItem) => {
      const entry = [...itemSlots].find(([, slotItem]) => toGalleryItemKey(slotItem) === toGalleryItemKey(item));

      if (entry) {
        setActiveCursor({ filterIdentity, index: entry[0] });
      }

      pickItem(item);
    },
    [filterIdentity, itemSlots, pickItem]
  );
  const handleVisibleRangeChange = useCallback(
    (range: { endIndexExclusive: number; startIndex: number }) => setVisibleRange?.(range),
    [setVisibleRange]
  );
  const handleClearSearch = useCallback(() => setSearchTerm(''), [setSearchTerm]);

  const openGallery = useCallback(() => {
    const boardId =
      data.selectedBoardId !== getGallerySelectedBoardId(galleryValues, data.boards) ? data.selectedBoardId : null;

    if (revealInGallery?.({ boardId, view: scope.galleryView })) {
      onClose();
    }
  }, [data.boards, data.selectedBoardId, galleryValues, onClose, revealInGallery, scope.galleryView]);

  const pickerNotice =
    scope.pane !== 'items'
      ? null
      : data.listing.status === 'more-error'
        ? { message: 'widgets.gallery.listingLoadMoreFailed', retryLabel: 'widgets.gallery.retryLoadingMoreItems' }
        : data.listing.status === 'stale-error'
          ? { message: 'widgets.gallery.listingRefreshFailed', retryLabel: 'widgets.gallery.retryLoadingItems' }
          : null;
  const showPickerNotice =
    pickerNotice !== null &&
    (data.sparseListing === undefined || [...pageStates.values()].some((pageState) => pageState.error !== null));
  // Counts of a list that failed to load would claim it is empty.
  const shouldHideStatus =
    isStale || (scope.pane === 'boards' ? isBoardListUnavailable : data.listing.status === 'error');
  const status = shouldHideStatus
    ? ''
    : getGalleryPickerStatus({
        accept,
        activeItem,
        isSearching,
        isWindowTruncated: data.isWindowTruncated,
        loadedCount: itemSlots.size,
        pane: scope.pane,
        remaining: getGalleryPickerRemaining(accept, selection),
        total,
        visibleBoardCount,
      })
        .map((part) => t(`widgets.gallery.picker.${part.kind}`, 'count' in part ? { count: part.count } : undefined))
        .join(' · ');

  const searchEndElement = useMemo(
    () => (
      <HStack flexShrink={0} gap="0">
        {scope.searchTerm ? (
          <CloseButton aria-label={t('common.clearSearch')} size="sm" onClick={handleClearSearch} />
        ) : null}
        {scope.pane === 'items' ? <GallerySearchHelp /> : null}
      </HStack>
    ),
    [handleClearSearch, scope.pane, scope.searchTerm, t]
  );
  const searchInputProps = useMemo(
    () =>
      scope.pane === 'boards'
        ? { 'aria-controls': boardsId }
        : showsGrid
          ? {
              'aria-activedescendant':
                resolvedActiveIndex >= 0 && !isStale
                  ? galleryPickerOptionId(listboxId, resolvedActiveIndex)
                  : undefined,
              'aria-controls': listboxId,
            }
          : undefined,
    [boardsId, isStale, listboxId, resolvedActiveIndex, scope.pane, showsGrid]
  );
  const searchLabel =
    scope.pane === 'boards'
      ? t('widgets.gallery.picker.findBoard')
      : t('widgets.gallery.picker.searchIn', { name: boardName });

  // Clips here rather than on the popover content, whose arrow sits outside its box.
  return (
    <Stack borderRadius="inherit" flex="1" gap="0" minH="0" overflow="hidden">
      <input {...uploadInputProps} />
      <HStack gap="1" minW="0" pt="2" px="2">
        <Button
          _expanded={BOARD_BUTTON_EXPANDED_PROPS}
          aria-controls={scope.pane === 'boards' ? boardsId : undefined}
          aria-expanded={scope.pane === 'boards'}
          aria-label={t('widgets.gallery.picker.changeBoardNamed', { name: boardName })}
          flex="1"
          justifyContent="flex-start"
          minW="0"
          ps="1"
          variant="ghost"
          onClick={togglePane}
        >
          {selectedBoard ? <BoardCover board={selectedBoard} /> : <BoardCoverIcon icon={ImageIcon} />}
          <MiddleTruncate fontWeight="600" minW="0" text={boardName} />
          <Icon
            as={scope.pane === 'boards' ? ChevronsDownUpIcon : ChevronsUpDownIcon}
            boxSize="3"
            color="fg.subtle"
            flexShrink={0}
          />
        </Button>
        <GalleryViewSegmentTabs
          activeView={scope.galleryView}
          board={selectedBoard}
          iconLabels
          idBase={idBase}
          onSelect={setView}
        />
        <Tooltip content={uploadTarget.label}>
          <IconButton
            aria-busy={isUploading || undefined}
            aria-label={uploadTarget.label}
            color="fg.muted"
            disabled={!uploadTarget.isAvailable || isUploading}
            variant="ghost"
            onClick={openUploadPicker}
          >
            <Icon as={UploadIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      </HStack>
      <Box pt="1.5" px="2">
        <GallerySearchField
          ref={attachSearchInput}
          ariaLabel={searchLabel}
          endElement={searchEndElement}
          inputProps={searchInputProps}
          placeholder={searchLabel}
          value={scope.searchTerm}
          onChange={setSearchTerm}
          onKeyDown={handleSearchKeyDown}
        />
      </Box>
      <Stack
        aria-labelledby={segmentTabsTabId(idBase, scope.galleryView)}
        flex="1"
        gap="0"
        id={segmentTabsPanelId(idBase)}
        minH="0"
        role="tabpanel"
      >
        {scope.pane === 'boards' && isBoardListUnavailable ? (
          <GalleryLoadErrorState
            read={data.boardsState}
            retryLabel={t('widgets.gallery.retryLoadingBoards')}
            title={t('widgets.gallery.boardsLoadFailed')}
            onFocusLost={focusSearch}
          />
        ) : scope.pane === 'boards' ? (
          <GalleryPickerBoards
            emptyMessage={
              data.boardsState.status === 'loading' ? t('common.loading') : t('widgets.gallery.noBoardsMatchSearch')
            }
            galleryView={scope.galleryView}
            groups={boardGroups}
            id={boardsId}
            label={t('widgets.gallery.picker.boardsLabel')}
            selectedBoardId={data.selectedBoardId}
            onExitTop={focusSearch}
            onSelect={handleSelectBoard}
          />
        ) : data.listing.status === 'error' ? (
          <GalleryLoadErrorState
            read={data.listing}
            retryLabel={t('widgets.gallery.retryLoadingItems')}
            title={t('widgets.gallery.listingLoadFailed')}
            onFocusLost={focusSearch}
          />
        ) : showsGrid ? (
          <GalleryPickerGrid
            activeIndex={resolvedActiveIndex}
            activeScope={filterIdentity}
            columnCount={columnCount}
            currentKey={currentKey}
            getTileState={getTileState}
            idBase={listboxId}
            isMultiple={isMultiple}
            isStale={isStale}
            itemSlots={itemSlots}
            label={t('widgets.gallery.picker.itemsLabel')}
            onActivate={handleActivate}
            onColumnCountChange={setColumnCount}
            onVisibleRangeChange={handleVisibleRangeChange}
            pageStates={pageStates}
            suppressInlineRetry={showPickerNotice}
            total={total}
          />
        ) : (
          <Stack align="center" color="fg.muted" gap="2" justify="center" minH="7rem" px="4" py="6">
            <Icon as={ImageIcon} boxSize="4" color="fg.subtle" />
            <Text fontSize="md" textAlign="center" textWrap="pretty">
              {isSearching
                ? t('widgets.gallery.noImagesMatch')
                : t('widgets.gallery.picker.empty', { name: boardName })}
            </Text>
            {!isSearching && uploadTarget.isAvailable ? (
              <Button disabled={isUploading} variant="outline" onClick={openUploadPicker}>
                <Icon as={UploadIcon} boxSize="3.5" />
                {t('widgets.gallery.picker.upload')}
              </Button>
            ) : null}
          </Stack>
        )}
        {showPickerNotice ? (
          <GalleryLoadNotice
            borderColor="border.subtle"
            borderTopWidth="1px"
            message={t(pickerNotice.message)}
            pe="1"
            ps="2"
            py="1"
            read={data.listing}
            retryLabel={t(pickerNotice.retryLabel)}
            onFocusLost={focusSearch}
          />
        ) : null}
      </Stack>
      <HStack borderColor="border.subtle" borderTopWidth="1px" gap="2" justify="space-between" pe="1" ps="2" py="1">
        <Text color="fg.subtle" fontSize="xs" fontVariantNumeric="tabular-nums" minW="0" role="status" truncate>
          {status}
        </Text>
        {isMultiple ? (
          <Button flexShrink={0} size="sm" variant="subtle" onClick={onClose}>
            {t('common.done')}
          </Button>
        ) : revealInGallery ? (
          <Button color="fg.muted" flexShrink={0} size="sm" variant="ghost" onClick={openGallery}>
            {t('widgets.gallery.picker.openGallery')}
            <Icon as={ExternalLinkIcon} boxSize="3" />
          </Button>
        ) : null}
      </HStack>
      <GalleryAnnouncer message={pickerNotice ? t(pickerNotice.message) : ''} />
    </Stack>
  );
};

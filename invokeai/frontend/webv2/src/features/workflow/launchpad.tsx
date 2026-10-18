import type { WorkflowLibraryCategory, WorkflowLibraryListItem } from '@features/workflow/data/api';
import type { WorkflowLibraryBrowseSnapshot } from '@features/workflow/data/libraryBrowseStore';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { HStack, Input, Stack } from '@chakra-ui/react';
import {
  ensureWorkflowLibraryBrowseLoaded,
  getWorkflowLibraryBrowseSnapshot,
  setWorkflowLibraryBrowseFilter,
  useWorkflowLibraryBrowseSelector,
} from '@features/workflow/data/libraryBrowseStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { WorkflowLibraryGrid } from './ui/library/WorkflowLibraryGrid';
import { WorkflowLibraryTagChips } from './ui/library/WorkflowLibraryTagChips';

/**
 * The library as a single-select picker for the Launchpad's New-project dialog: search, Browse/Yours, tag chips and
 * the card grid, without the editor dialog's actions, detail panel or This-project view. The browse store is shared
 * with the editor dialog, so filters carry over between the two.
 */

const SEARCH_DEBOUNCE_MS = 300;

const TAB_ITEMS: ReadonlyArray<{ labelKey: string; value: WorkflowLibraryCategory }> = [
  { labelKey: 'workflowLibrary.browse', value: 'default' },
  { labelKey: 'workflowLibrary.yours', value: 'user' },
];

/** Flat and shallow-comparable, so unrelated store patches do not re-render the picker. */
const selectBrowseView = (snapshot: WorkflowLibraryBrowseSnapshot) => ({
  category: snapshot.filter.category,
  entries: snapshot.entries,
  error: snapshot.error,
  status: snapshot.status,
  tag: snapshot.filter.tag,
  tagCounts: snapshot.tagCounts,
});

export interface WorkflowLibraryPickerProps {
  selectedWorkflowId: string | null;
  onSelect: (item: WorkflowLibraryListItem) => void;
  /** A double-click or Enter on a card: the pick is final. */
  onOpen: (item: WorkflowLibraryListItem) => void;
}

export const WorkflowLibraryPicker = ({ onOpen, onSelect, selectedWorkflowId }: WorkflowLibraryPickerProps) => {
  const { t } = useTranslation();
  const { category, entries, error, status, tag, tagCounts } = useWorkflowLibraryBrowseSelector(selectBrowseView);
  const [searchInput, setSearchInput] = useState('');
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tabsIdBase = useId();

  // Start unfiltered (the store is shared with the editor dialog) and load the first page; an account without saved
  // workflows starts on Browse, as the editor dialog does.
  useMountEffect(() => {
    if (getWorkflowLibraryBrowseSnapshot().filter.search) {
      setWorkflowLibraryBrowseFilter({ search: '' });
    }

    void ensureWorkflowLibraryBrowseLoaded().then(() => {
      const { filter, userTotal } = getWorkflowLibraryBrowseSnapshot();

      if (userTotal === 0 && filter.category === 'user' && !filter.search) {
        setWorkflowLibraryBrowseFilter({ category: 'default', tag: null });
      }
    });

    return () => {
      if (searchTimerRef.current !== null) {
        clearTimeout(searchTimerRef.current);
      }
    };
  });

  const handleSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const { value } = event.currentTarget;

    setSearchInput(value);

    if (searchTimerRef.current !== null) {
      clearTimeout(searchTimerRef.current);
    }

    searchTimerRef.current = setTimeout(() => setWorkflowLibraryBrowseFilter({ search: value }), SEARCH_DEBOUNCE_MS);
  }, []);
  const handleTabChange = useCallback(
    (value: WorkflowLibraryCategory) => {
      // Chips belong to a category, so changing it clears the tag.
      if (category !== value) {
        setWorkflowLibraryBrowseFilter({ category: value, tag: null });
      }
    },
    [category]
  );
  const handleTagSelect = useCallback((nextTag: string | null) => setWorkflowLibraryBrowseFilter({ tag: nextTag }), []);

  const findItem = useCallback(
    (workflowId: string) =>
      getWorkflowLibraryBrowseSnapshot().entries.find((entry) => entry.item.workflow_id === workflowId)?.item,
    []
  );
  const handleSelect = useCallback(
    (workflowId: string) => {
      const item = findItem(workflowId);

      if (item) {
        onSelect(item);
      }
    },
    [findItem, onSelect]
  );
  const handleOpen = useCallback(
    (workflowId: string) => {
      const item = findItem(workflowId);

      if (item) {
        onOpen(item);
      }
    },
    [findItem, onOpen]
  );
  // Cards open on double-click; Enter on a focused card is its keyboard equivalent.
  const handleGridKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const workflowId = (event.target as HTMLElement).closest<HTMLElement>('[data-workflow-card]')?.dataset
        .workflowCard;

      if (event.key === 'Enter' && workflowId) {
        event.preventDefault();
        handleOpen(workflowId);
      }
    },
    [handleOpen]
  );

  const tabItems = useMemo(() => TAB_ITEMS.map((item) => ({ id: item.value, label: t(item.labelKey) })), [t]);

  return (
    <Stack flex="1" gap="3" minH="0" minW="0">
      <HStack gap="3" minW="0">
        <Input
          aria-label={t('workflowLibrary.searchPlaceholder')}
          flex="1"
          minW="0"
          placeholder={t('workflowLibrary.searchPlaceholder')}
          type="search"
          value={searchInput}
          onChange={handleSearchChange}
        />
        <SegmentTabs
          activeId={category}
          ariaLabel={t('workflowLibrary.title')}
          idBase={tabsIdBase}
          isCompact
          tabs={tabItems}
          onSelect={handleTabChange}
        />
      </HStack>
      <WorkflowLibraryTagChips selectedTag={tag} tagCounts={tagCounts} onSelect={handleTagSelect} />
      <Stack
        aria-labelledby={segmentTabsTabId(tabsIdBase, category)}
        display="flex"
        flex="1"
        id={segmentTabsPanelId(tabsIdBase)}
        minH="0"
        minW="0"
        role="tabpanel"
        onKeyDown={handleGridKeyDown}
      >
        <WorkflowLibraryGrid
          entries={entries}
          error={error}
          selectedWorkflowId={selectedWorkflowId}
          status={status}
          onOpen={handleOpen}
          onSelect={handleSelect}
        />
      </Stack>
    </Stack>
  );
};

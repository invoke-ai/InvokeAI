import type { WorkflowLibraryBrowseSnapshot, WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { Box, HStack, Spinner, Text } from '@chakra-ui/react';
import { loadNextWorkflowLibraryPage } from '@features/workflow/data/libraryBrowseStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Scrollable } from '@platform/ui';
import { useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { WorkflowLibraryCard } from './WorkflowLibraryCard';

/**
 * The scrolling card grid. Paging is server side and append-only: reaching the
 * bottom of the accumulated pages asks the browse store for the next one,
 * which no-ops once the last page has landed.
 */

/** How close to the bottom (in viewports) counts as "fetch the next page". */
const NEAR_BOTTOM_VIEWPORTS = 1.5;
const GRID_TEMPLATE_COLUMNS = 'repeat(2, minmax(0, 1fr))';
const NO_MISSING_COUNTS: ReadonlyMap<string, number> = new Map();

export interface WorkflowLibraryGridProps {
  entries: readonly WorkflowLibraryEntry[];
  error: string | null;
  /** Missing-model counts by workflow id; absent ids render no badge. */
  missingCounts?: ReadonlyMap<string, number>;
  selectedWorkflowId: string | null;
  status: WorkflowLibraryBrowseSnapshot['status'];
  onOpen: (workflowId: string) => void;
  onSelect: (workflowId: string) => void;
}

/**
 * A refresh landing while an append is in flight can publish the same row
 * twice for one frame. Duplicate React keys would corrupt the reconciliation,
 * so the grid keeps the first occurrence and drops the rest.
 */
const dedupeByWorkflowId = (entries: readonly WorkflowLibraryEntry[]): WorkflowLibraryEntry[] => {
  const seen = new Set<string>();
  const unique: WorkflowLibraryEntry[] = [];

  for (const entry of entries) {
    if (!seen.has(entry.item.workflow_id)) {
      seen.add(entry.item.workflow_id);
      unique.push(entry);
    }
  }

  return unique;
};

export const WorkflowLibraryGrid = ({
  entries,
  error,
  missingCounts = NO_MISSING_COUNTS,
  onOpen,
  onSelect,
  selectedWorkflowId,
  status,
}: WorkflowLibraryGridProps) => {
  const { t } = useTranslation();
  const viewportRef = useRef<HTMLDivElement | null>(null);

  // Infinite scroll reads geometry off the scrolling element itself, which is
  // inside `Scrollable`, so the listener is registered directly rather than
  // through a React `onScroll` prop (scroll events do not bubble).
  useMountEffect(() => {
    const viewport = viewportRef.current;

    if (!viewport) {
      return;
    }

    const handleScroll = () => {
      const remaining = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;

      if (remaining <= viewport.clientHeight * NEAR_BOTTOM_VIEWPORTS) {
        loadNextWorkflowLibraryPage();
      }
    };

    viewport.addEventListener('scroll', handleScroll, { passive: true });

    return () => viewport.removeEventListener('scroll', handleScroll);
  });

  const visibleEntries = useMemo(() => dedupeByWorkflowId(entries), [entries]);
  const hasEntries = visibleEntries.length > 0;

  return (
    <Scrollable flex="1" label={t('workflowLibrary.title')} minH="0" minW="0" viewportRef={viewportRef}>
      <Box display="flex" flexDirection="column" gap="3" minW="0" pr="2" w="full">
        {hasEntries ? (
          <Box display="grid" gap="3" gridTemplateColumns={GRID_TEMPLATE_COLUMNS} minW="0" w="full">
            {visibleEntries.map((entry) => (
              <WorkflowLibraryCard
                key={entry.item.workflow_id}
                entry={entry}
                isSelected={entry.item.workflow_id === selectedWorkflowId}
                missingCount={missingCounts.get(entry.item.workflow_id) ?? 0}
                onOpen={onOpen}
                onSelect={onSelect}
              />
            ))}
          </Box>
        ) : null}
        {!hasEntries && status === 'loading' ? (
          <Text color="fg.subtle" fontSize="xs" py="6" textAlign="center">
            {t('workflowLibrary.loading')}
          </Text>
        ) : null}
        {!hasEntries && status !== 'loading' ? (
          <Text color="fg.subtle" fontSize="xs" py="6" textAlign="center">
            {error ?? t('workflowLibrary.empty')}
          </Text>
        ) : null}
        {hasEntries && status === 'loadingMore' ? (
          <HStack color="fg.subtle" gap="2" justify="center" py="2">
            <Spinner size="xs" />
            <Text fontSize="2xs">{t('workflowLibrary.loadingMore')}</Text>
          </HStack>
        ) : null}
        {hasEntries && error ? (
          // A failed page append never blanks the pages already loaded.
          <Text color="fg.subtle" fontSize="2xs" py="2" textAlign="center">
            {error}
          </Text>
        ) : null}
      </Box>
    </Scrollable>
  );
};

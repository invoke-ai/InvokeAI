import type { WorkflowLibraryBrowseSnapshot, WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { Box, HStack, Spinner, Text } from '@chakra-ui/react';
import { loadNextWorkflowLibraryPage } from '@features/workflow/data/libraryBrowseStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Scrollable } from '@platform/ui';
import { useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { WorkflowLibraryCard, type WorkflowCardMenuAnchor, type WorkflowLibraryCardProps } from './WorkflowLibraryCard';

/** How close to the bottom (in viewports) counts as "fetch the next page". */
const NEAR_BOTTOM_VIEWPORTS = 1.5;
const GRID_TEMPLATE_COLUMNS = 'repeat(3, minmax(0, 1fr))';
const NO_MISSING_COUNTS: ReadonlyMap<string, number> = new Map();

export interface WorkflowLibraryGridProps {
  /** The project's active workflow, marked on its card in the This-project view. */
  activeWorkflowId?: string | null;
  entries: readonly WorkflowLibraryEntry[];
  error: string | null;
  /** What has the actions menu open, so that tile's control can name it. */
  openMenuAnchor?: WorkflowCardMenuAnchor | null;
  /** Missing-model counts by workflow id; absent ids render no badge. */
  missingCounts?: ReadonlyMap<string, number>;
  selectedWorkflowId: string | null;
  status: WorkflowLibraryBrowseSnapshot['status'];
  onContextMenu: WorkflowLibraryCardProps['onContextMenu'];
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
  activeWorkflowId = null,
  entries,
  error,
  missingCounts = NO_MISSING_COUNTS,
  openMenuAnchor = null,
  onContextMenu,
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
      // A grid that doesn't overflow cannot have been scrolled near its
      // bottom by anyone; a scroll event there is measurement noise.
      if (viewport.scrollHeight <= viewport.clientHeight) {
        return;
      }

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
  // Idle precedes the first open-time load; render loading rather than an empty result.
  const isPending = status === 'idle' || status === 'loading';

  return (
    <Scrollable flex="1" label={t('workflowLibrary.title')} minH="0" minW="0" viewportRef={viewportRef}>
      <Box display="flex" flexDirection="column" gap="3" minW="0" pr="2" w="full">
        {hasEntries ? (
          <Box display="grid" gap="3" gridTemplateColumns={GRID_TEMPLATE_COLUMNS} minW="0" w="full">
            {visibleEntries.map((entry) => (
              <WorkflowLibraryCard
                key={entry.item.workflow_id}
                entry={entry}
                isActive={entry.item.workflow_id === activeWorkflowId}
                isSelected={entry.item.workflow_id === selectedWorkflowId}
                menuOpenedBy={
                  openMenuAnchor?.workflowId === entry.item.workflow_id
                    ? openMenuAnchor.kind === 'trigger'
                      ? 'button'
                      : 'card'
                    : null
                }
                missingCount={missingCounts.get(entry.item.workflow_id) ?? 0}
                onContextMenu={onContextMenu}
                onOpen={onOpen}
                onSelect={onSelect}
              />
            ))}
          </Box>
        ) : null}
        {!hasEntries && isPending ? (
          <Text color="fg.subtle" fontSize="xs" py="6" textAlign="center">
            {t('workflowLibrary.loading')}
          </Text>
        ) : null}
        {!hasEntries && !isPending ? (
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

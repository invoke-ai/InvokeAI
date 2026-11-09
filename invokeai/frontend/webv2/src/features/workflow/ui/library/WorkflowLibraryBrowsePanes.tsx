import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryBrowseSnapshot, WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { WorkflowLibraryTab } from '@features/workflow/ui/workflowUiStore';

import { HStack, type SystemStyleObject, Text } from '@chakra-ui/react';
import { useAuthSession } from '@features/identity';
import {
  selectWorkflowLibraryEntry,
  setWorkflowLibrarySelectedDemand,
  useWorkflowLibraryBrowseSelector,
} from '@features/workflow/data/libraryBrowseStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowCardContextMenuHandler, WorkflowCardMenuAnchor } from './WorkflowLibraryCard';

import { WORKFLOW_DETAIL_PANES_CSS } from './WorkflowDetailSheet';
import { WorkflowLibraryDetailPanel } from './WorkflowLibraryDetailPanel';
import { WorkflowLibraryGrid } from './WorkflowLibraryGrid';
import { WORKFLOW_LIBRARY_WIDE } from './WorkflowLibraryRail';
import { useWorkflowLibraryMissingCounts } from './WorkflowRequirementsList';

/**
 * The template places' result line, grid and detail rail. They alone subscribe to the listing's entries, so range
 * loads and enrichment results re-render this subtree, never the dialog shell, its rail or its toolbar.
 */

/**
 * The card grid and the detail rail, in both views. In the wide layout their box dissolves into the dialog body's grid
 * (see `WorkflowLibraryDialog`): the cards take the centre column under the toolbar, and the detail rail becomes a
 * flush column from under the header to the dialog's bottom edge, separated by a 1px border rather than drawn as a
 * card. The narrow layout keeps the sheet's own grid (`WORKFLOW_DETAIL_PANES_CSS`).
 */
export const WORKFLOW_LIBRARY_PANES_CSS: SystemStyleObject = {
  ...WORKFLOW_DETAIL_PANES_CSS,
  [WORKFLOW_LIBRARY_WIDE]: {
    display: 'contents',
    '& > :first-child': { gridArea: 'panes', mb: '3', minH: '0' },
    // Per side, not the shorthand: `borderWidth: 0` after `borderStartWidth` would reset the start border too.
    '& > :last-child': {
      borderBottomWidth: '0',
      borderEndWidth: '0',
      borderRadius: '0',
      borderStartWidth: '1px',
      borderTopWidth: '0',
      gridArea: 'detail',
      minH: '0',
    },
  },
};

/** Flat and shallow-comparable, so place totals, tag counts and scroll anchors do not re-render the panes. */
const selectPanes = (snapshot: WorkflowLibraryBrowseSnapshot) => ({
  viewKey: JSON.stringify(snapshot.filter),
  entries: snapshot.entries,
  startIndex: snapshot.startIndex ?? 0,
  filter: snapshot.filter,
  total: snapshot.total,
  selectedEntry: snapshot.selectedEntry,
  error: snapshot.error,
  status: snapshot.status,
});

export interface WorkflowLibraryResultLineProps {
  /** Matching workflows; null while unknown. */
  count: number | null;
  /** The checked tags as a localized "a or b" list; empty when none. */
  tagsText: string;
}

/**
 * "N workflows · tags" above the cards. Nothing about requirements: on a server-paginated place only the enriched
 * window is known, so any aggregate would change while scrolling.
 */
export const WorkflowLibraryResultLine = memo(function WorkflowLibraryResultLine({
  count,
  tagsText,
}: WorkflowLibraryResultLineProps) {
  const { t } = useTranslation();

  return (
    <Text
      aria-live="polite"
      color="fg.muted"
      data-workflow-library-results
      flexShrink={0}
      fontSize="sm"
      minH="4"
      minW="0"
      truncate
    >
      {count === null
        ? null
        : tagsText
          ? t('workflowLibrary.resultCountWithTags', { count, tags: tagsText })
          : t('workflowLibrary.resultCount', { count })}
    </Text>
  );
});

export interface WorkflowLibraryBrowsePanesProps {
  /** Whether the dialog is open; selected-detail demand runs only then. */
  isOpen: boolean;
  tab: WorkflowLibraryTab;
  tagsText: string;
  emptyMessage: string | undefined;
  contextMenuPoint: WorkflowCardMenuAnchor | null;
  contextMenuTriggerId: string | null;
  /** The user's explicit choice; the panes fall back to the selected pin or the first resident entry. */
  selectedWorkflowId: string | null;
  removedWorkflowId: string | null;
  projectWorkflows: readonly ProjectWorkflowEntry[];
  onClose: () => void;
  onContextMenu: WorkflowCardContextMenuHandler;
  onContextMenuClose: () => void;
  /** Receives the workflow the detail rail showed when it was deleted. */
  onDeleted: (workflowId: string | null) => void;
  onOpen: (workflowId: string) => void;
  onOpenItem: (item: WorkflowLibraryListItem) => void;
  onCardPreview: (workflowId: string) => void;
  onPreview: (entry: WorkflowLibraryEntry) => void;
  onSelect: (workflowId: string) => void;
}

export const WorkflowLibraryBrowsePanes = memo(function WorkflowLibraryBrowsePanes({
  contextMenuPoint,
  contextMenuTriggerId,
  emptyMessage,
  isOpen,
  projectWorkflows,
  removedWorkflowId,
  selectedWorkflowId,
  tab,
  tagsText,
  onCardPreview,
  onClose,
  onContextMenu,
  onContextMenuClose,
  onDeleted,
  onOpen,
  onOpenItem,
  onPreview,
  onSelect,
}: WorkflowLibraryBrowsePanesProps) {
  const { entries, error, filter, selectedEntry, startIndex, status, total, viewKey } =
    useWorkflowLibraryBrowseSelector(selectPanes);
  const authSession = useAuthSession();
  const missingCounts = useWorkflowLibraryMissingCounts(entries);

  // Derive a fallback selection when filtering/deletion removes the selected row.
  const desiredWorkflowId = selectedWorkflowId ?? selectedEntry?.item.workflow_id ?? null;
  const activeWorkflowId =
    selectedEntry?.item.workflow_id === desiredWorkflowId ||
    entries.some((entry) => entry.item.workflow_id === desiredWorkflowId)
      ? desiredWorkflowId
      : (entries[0]?.item.workflow_id ?? null);
  const activeEntry = selectWorkflowLibraryEntry({ entries, selectedEntry }, activeWorkflowId) ?? null;
  const isCurrentLibraryOwner =
    authSession.phase === 'ready' &&
    (authSession.multiuserEnabled
      ? Boolean(activeEntry?.item.user_id && activeEntry.item.user_id === authSession.user?.user_id)
      : true);
  const canEditLibraryEntry = activeEntry?.item.category === 'user' && isCurrentLibraryOwner;
  const isReadOnlyLibraryEntry = activeEntry?.item.category === 'user' && !isCurrentLibraryOwner;
  const canChangeVisibility =
    tab === 'user' &&
    authSession.phase === 'ready' &&
    authSession.multiuserEnabled &&
    activeEntry?.item.category === 'user' &&
    isCurrentLibraryOwner;
  const browseWindow = useMemo(() => ({ startIndex, total, filter }), [startIndex, total, filter]);
  const handleDeleted = useCallback(() => onDeleted(activeWorkflowId), [activeWorkflowId, onDeleted]);

  return (
    <>
      <WorkflowLibraryResultLine count={status === 'loaded' ? total : null} tagsText={tagsText} />
      <HStack css={WORKFLOW_LIBRARY_PANES_CSS} align="stretch" flex="1" gap="3" minH="0" minW="0">
        {isOpen ? (
          <SelectedDemand key={`${authSession.user?.user_id}:${viewKey}:${activeWorkflowId}`} id={activeWorkflowId} />
        ) : null}
        <WorkflowLibraryGrid
          key={`${authSession.user?.user_id}:${viewKey}`}
          window={browseWindow}
          removedWorkflowId={removedWorkflowId}
          emptyMessage={emptyMessage}
          entries={entries}
          error={error}
          missingCounts={missingCounts}
          openMenuAnchor={contextMenuPoint}
          showVisibilityStatus={authSession.multiuserEnabled}
          selectedWorkflowId={activeWorkflowId}
          status={status}
          onContextMenu={onContextMenu}
          onOpen={onOpen}
          onPreview={onCardPreview}
          onSelect={onSelect}
        />
        <WorkflowLibraryDetailPanel
          contextMenuPoint={contextMenuPoint}
          contextMenuTriggerId={contextMenuTriggerId}
          entry={activeEntry}
          canChangeVisibility={canChangeVisibility}
          isReadOnly={isReadOnlyLibraryEntry || !canEditLibraryEntry}
          projectWorkflows={projectWorkflows}
          onClose={onClose}
          onContextMenuClose={onContextMenuClose}
          onDeleted={handleDeleted}
          onDuplicated={onSelect}
          onOpen={onOpenItem}
          onPreview={onPreview}
        />
      </HStack>
    </>
  );
});

let selectedLease = 0;
const SelectedDemand = ({ id }: { id: string | null }) => {
  useMountEffect(() => {
    const lease = ++selectedLease;
    setWorkflowLibrarySelectedDemand(id);
    return () => {
      queueMicrotask(() => {
        if (lease === selectedLease) {
          setWorkflowLibrarySelectedDemand(null);
        }
      });
    };
  });
  return null;
};

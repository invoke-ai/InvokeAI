import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryBrowseSnapshot, WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { WorkflowPreviewView } from '@features/workflow/ui/graph-preview/WorkflowPreviewStage';
import type { WorkflowLibraryLayout } from '@features/workflow/ui/WorkflowUiContext';
import type { WorkflowLibraryTab } from '@features/workflow/ui/workflowUiStore';

import { HStack, type SystemStyleObject, Text } from '@chakra-ui/react';
import { useAuthSession } from '@features/identity';
import {
  getWorkflowLibraryBrowseSnapshot,
  loadWorkflowLibraryRange,
  selectWorkflowLibraryEntry,
  setWorkflowLibrarySelectedDemand,
  useWorkflowLibraryBrowseSelector,
} from '@features/workflow/data/libraryBrowseStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { memo, type RefObject, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowCardContextMenuHandler, WorkflowCardMenuAnchor } from './WorkflowLibraryCard';
import type { WorkflowContextMenuTarget } from './WorkflowLibraryContextMenu';
import type { WorkflowLibraryListTrailingField } from './WorkflowLibraryList';
import type { WorkflowLibraryPreviewStageHandle, WorkflowLibraryPreviewStatus } from './WorkflowLibraryPreviewStage';

import { useWorkflowLibraryDuplicateGuard } from './useWorkflowLibraryEntryActions';
import { WORKFLOW_DETAIL_PANES_CSS } from './WorkflowDetailSheet';
import { WorkflowLibraryContextMenu } from './WorkflowLibraryContextMenu';
import { WorkflowLibraryDetailPanel } from './WorkflowLibraryDetailPanel';
import { WorkflowLibraryGrid } from './WorkflowLibraryGrid';
import { WorkflowLibraryList } from './WorkflowLibraryList';
import { WorkflowLibraryPreviewStage } from './WorkflowLibraryPreviewStage';
import { WORKFLOW_LIBRARY_WIDE } from './WorkflowLibraryRail';
import { useWorkflowLibraryMissingCounts } from './WorkflowRequirementsList';

/**
 * The template places' result line, grid or list, detail rail and card actions menu. They alone subscribe to the
 * listing's entries, so range loads and enrichment results re-render this subtree, never the dialog shell, its rail or
 * its toolbar.
 */

/**
 * The card grid (or the list) and the detail rail, in both views. In the wide layout their box dissolves into the
 * dialog body's grid (see `WorkflowLibraryDialog`): the cards take the centre column under the toolbar, and the detail
 * rail becomes a flush column from under the header to the dialog's bottom edge, separated by a 1px border rather than
 * drawn as a card. The narrow layout keeps the sheet's own grid (`WORKFLOW_DETAIL_PANES_CSS`).
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

type AuthSession = ReturnType<typeof useAuthSession>;

/**
 * What the account may do to a library entry: only its own templates (on a multi-user server, the ones it owns) are
 * writable, and visibility changes only from My library on a multi-user server.
 */
const getLibraryEntryAccess = (
  entry: WorkflowLibraryEntry | null | undefined,
  session: AuthSession,
  tab: WorkflowLibraryTab
) => {
  const isOwner =
    session.phase === 'ready' &&
    (session.multiuserEnabled ? Boolean(entry?.item.user_id && entry.item.user_id === session.user?.user_id) : true);
  const isWritable = entry?.item.category === 'user' && isOwner;

  return { canChangeVisibility: isWritable && tab === 'user' && session.multiuserEnabled, isReadOnly: !isWritable };
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
  layout: WorkflowLibraryLayout;
  tab: WorkflowLibraryTab;
  /** The date the list's rows show, from the current order. */
  trailingField: WorkflowLibraryListTrailingField;
  tagsText: string;
  emptyMessage: string | undefined;
  /** Where the card actions menu is anchored; null while it is closed. */
  contextMenuAnchor: WorkflowCardMenuAnchor | null;
  /** The workflow the card actions menu was raised for, kept through its close. */
  contextMenuTarget: WorkflowContextMenuTarget | null;
  /** The user's explicit choice; the panes fall back to the selected pin or the first resident entry. */
  selectedWorkflowId: string | null;
  removedWorkflowId: string | null;
  projectWorkflows: readonly ProjectWorkflowEntry[];
  onClose: () => void;
  onContextMenu: WorkflowCardContextMenuHandler;
  onContextMenuClose: () => void;
  /** Receives the deleted workflow, from the rail or a card's menu. */
  onDeleted: (workflowId: string) => void;
  onOpen: (workflowId: string) => void;
  onOpenItem: (item: WorkflowLibraryListItem) => void;
  /** Enters preview mode on a workflow, selecting it; on a given view, else the one last shown. */
  onPreview: (workflowId: string, view?: WorkflowPreviewView) => void;
  onSelect: (workflowId: string) => void;
  preview: WorkflowLibraryPanesPreview;
  /** The stage's handle, through which the dialog's Escape reaches it. */
  previewStageRef: RefObject<WorkflowLibraryPreviewStageHandle | null>;
}

/** Preview mode, which the shell owns: the stage replaces the collection while the detail rail stays. */
export interface WorkflowLibraryPanesPreview {
  isActive: boolean;
  /** The place the stage's back button returns to. */
  placeLabel: string;
  onLeave: (workflowId: string | null) => void;
  view: WorkflowPreviewView;
  onViewChange: (view: WorkflowPreviewView) => void;
}

/** What the stage shows for an entry: its enriched document, or why there is none yet. */
const getLibraryPreviewStatus = (entry: WorkflowLibraryEntry | null): WorkflowLibraryPreviewStatus =>
  entry?.enrichment.status === 'ready'
    ? { document: entry.enrichment.document, kind: 'ready' }
    : entry?.enrichment.status === 'error'
      ? { kind: 'error', message: entry.enrichment.message }
      : { kind: 'pending' };

export const WorkflowLibraryBrowsePanes = memo(function WorkflowLibraryBrowsePanes({
  contextMenuAnchor,
  contextMenuTarget,
  emptyMessage,
  isOpen,
  layout,
  projectWorkflows,
  removedWorkflowId,
  selectedWorkflowId,
  tab,
  tagsText,
  trailingField,
  preview,
  previewStageRef,
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
  const { t } = useTranslation();
  const missingCounts = useWorkflowLibraryMissingCounts(entries);

  // Derive a fallback selection when filtering/deletion removes the selected row.
  const desiredWorkflowId = selectedWorkflowId ?? selectedEntry?.item.workflow_id ?? null;
  const activeWorkflowId =
    selectedEntry?.item.workflow_id === desiredWorkflowId ||
    entries.some((entry) => entry.item.workflow_id === desiredWorkflowId)
      ? desiredWorkflowId
      : (entries[0]?.item.workflow_id ?? null);
  const activeEntry = selectWorkflowLibraryEntry({ entries, selectedEntry }, activeWorkflowId) ?? null;
  const activeAccess = getLibraryEntryAccess(activeEntry, authSession, tab);
  // The menu acts on the card it was raised for, resident or the selected pin; once that leaves, it has nothing to
  // act on and stays closed (the shell clears its anchor).
  const menuEntry = contextMenuTarget
    ? (selectWorkflowLibraryEntry({ entries, selectedEntry }, contextMenuTarget.workflowId) ?? null)
    : null;
  const browseWindow = useMemo(() => ({ startIndex, total, filter }), [startIndex, total, filter]);
  const duplicateGuard = useWorkflowLibraryDuplicateGuard();

  // The selection's server position, while its page is resident; a pinned selection outside the window has none.
  const residentIndex = entries.findIndex((entry) => entry.item.workflow_id === activeWorkflowId);
  const position = residentIndex >= 0 ? startIndex + residentIndex : null;
  const previewStatus = useMemo(() => getLibraryPreviewStatus(activeEntry), [activeEntry]);
  const stepPreview = useCallback(
    (delta: -1 | 1, isCurrent: () => boolean) => {
      const target = position === null ? -1 : position + delta;

      if (target < 0 || target >= total) {
        return;
      }

      const resident = entries[target - startIndex];

      if (resident) {
        onSelect(resident.item.workflow_id);
        return;
      }

      // Past the resident window: read that position as scrolling to it would, then select what arrived there.
      void loadWorkflowLibraryRange(target, target).then((ready) => {
        const snapshot = getWorkflowLibraryBrowseSnapshot();
        const loaded = snapshot.entries[target - (snapshot.startIndex ?? 0)];

        if (ready && loaded && isCurrent()) {
          onSelect(loaded.item.workflow_id);
        }
      });
    },
    [entries, onSelect, position, startIndex, total]
  );

  const stepper = useMemo(
    () =>
      preview.isActive
        ? {
            canStepNext: position !== null && position < total - 1,
            canStepPrevious: position !== null && position > 0,
            onStep: stepPreview,
            position,
            total,
          }
        : null,
    [position, preview.isActive, stepPreview, total]
  );

  return (
    <>
      <WorkflowLibraryResultLine count={status === 'loaded' ? total : null} tagsText={tagsText} />
      <HStack css={WORKFLOW_LIBRARY_PANES_CSS} align="stretch" flex="1" gap="3" minH="0" minW="0">
        {isOpen ? (
          <SelectedDemand key={`${authSession.user?.user_id}:${viewKey}:${activeWorkflowId}`} id={activeWorkflowId} />
        ) : null}
        {layout === 'list' ? (
          <WorkflowLibraryList
            key={`${authSession.user?.user_id}:${viewKey}`}
            window={browseWindow}
            removedWorkflowId={removedWorkflowId}
            emptyMessage={emptyMessage}
            entries={entries}
            error={error}
            isHidden={preview.isActive}
            missingCounts={missingCounts}
            openMenuAnchor={contextMenuAnchor}
            showVisibilityStatus={authSession.multiuserEnabled}
            selectedWorkflowId={activeWorkflowId}
            status={status}
            onContextMenu={onContextMenu}
            onOpen={onOpen}
            onPreview={onPreview}
            onSelect={onSelect}
            trailingField={trailingField}
          />
        ) : (
          <WorkflowLibraryGrid
            key={`${authSession.user?.user_id}:${viewKey}`}
            window={browseWindow}
            removedWorkflowId={removedWorkflowId}
            emptyMessage={emptyMessage}
            entries={entries}
            error={error}
            isHidden={preview.isActive}
            missingCounts={missingCounts}
            openMenuAnchor={contextMenuAnchor}
            showVisibilityStatus={authSession.multiuserEnabled}
            selectedWorkflowId={activeWorkflowId}
            status={status}
            onContextMenu={onContextMenu}
            onOpen={onOpen}
            onPreview={onPreview}
            onSelect={onSelect}
          />
        )}
        {preview.isActive ? (
          <WorkflowLibraryPreviewStage
            ref={previewStageRef}
            name={activeEntry?.item.name || t('workflowLibrary.untitled')}
            placeLabel={preview.placeLabel}
            status={previewStatus}
            workflowId={activeWorkflowId}
            view={preview.view}
            onLeave={preview.onLeave}
            onViewChange={preview.onViewChange}
          />
        ) : null}
        <WorkflowLibraryDetailPanel
          entry={activeEntry}
          canChangeVisibility={activeAccess.canChangeVisibility}
          duplicateGuard={duplicateGuard}
          isReadOnly={activeAccess.isReadOnly}
          projectWorkflows={projectWorkflows}
          onClose={onClose}
          onDeleted={onDeleted}
          onDuplicated={onSelect}
          onOpen={onOpenItem}
          onPreview={onPreview}
          stepper={stepper}
        />
      </HStack>
      <WorkflowLibraryContextMenu
        anchor={contextMenuAnchor}
        duplicateGuard={duplicateGuard}
        entry={menuEntry}
        isReadOnly={getLibraryEntryAccess(menuEntry, authSession, tab).isReadOnly}
        projectWorkflows={projectWorkflows}
        target={contextMenuTarget}
        onClose={onClose}
        onDeleted={onDeleted}
        onDuplicated={onSelect}
        onMenuClose={onContextMenuClose}
        onOpen={onOpenItem}
      />
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

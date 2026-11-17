import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryBrowseSnapshot, WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { WorkflowPreviewView } from '@features/workflow/ui/graph-preview/WorkflowPreviewStage';
import type { WorkflowLibraryLayout } from '@features/workflow/ui/WorkflowUiContext';

import { HStack, type SystemStyleObject, Text } from '@chakra-ui/react';
import { useAuthSession } from '@features/identity';
import {
  getWorkflowLibraryBrowseSnapshot,
  loadWorkflowLibraryRange,
  selectWorkflowLibraryEntry,
  setWorkflowLibrarySelectedDemand,
  useWorkflowLibraryBrowseSelector,
} from '@features/workflow/data/libraryBrowseStore';
import { workflowUiStore, type WorkflowLibraryTab } from '@features/workflow/ui/workflowUiStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { memo, type RefObject, useCallback, useMemo, useRef } from 'react';
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

/** Only this subtree subscribes to listing entries, so range loads and enrichment never re-render the dialog shell. */

/** Wide, this box dissolves into the dialog body's grid; narrow, it keeps the sheet's own grid. */
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

/** Only the account's own templates are writable; visibility changes only from My library on a multi-user server. */
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
  tagsText: string;
}

/** No requirement aggregate: a server-paginated place knows only its enriched window, which changes while scrolling. */
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
  trailingField: WorkflowLibraryListTrailingField;
  tagsText: string;
  emptyMessage: string | undefined;
  contextMenuAnchor: WorkflowCardMenuAnchor | null;
  contextMenuTarget: WorkflowContextMenuTarget | null;
  /** The user's explicit choice; the panes fall back to the selected pin or the first resident entry. */
  selectedWorkflowId: string | null;
  removedWorkflowId: string | null;
  projectWorkflows: readonly ProjectWorkflowEntry[];
  onClose: () => void;
  onContextMenu: WorkflowCardContextMenuHandler;
  onContextMenuClose: () => void;
  onDeleted: (workflowId: string) => void;
  onOpen: (workflowId: string) => void;
  onOpenItem: (item: WorkflowLibraryListItem) => void;
  onPreview: (workflowId: string, view?: WorkflowPreviewView) => void;
  onSelect: (workflowId: string) => void;
  preview: WorkflowLibraryPanesPreview;
  previewStageRef: RefObject<WorkflowLibraryPreviewStageHandle | null>;
}

export interface WorkflowLibraryPanesPreview {
  isActive: boolean;
  placeLabel: string;
  onLeave: (workflowId: string | null) => void;
  view: WorkflowPreviewView;
  onViewChange: (view: WorkflowPreviewView) => void;
}

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

  const desiredWorkflowId = selectedWorkflowId ?? selectedEntry?.item.workflow_id ?? null;
  const activeWorkflowId =
    selectedEntry?.item.workflow_id === desiredWorkflowId ||
    entries.some((entry) => entry.item.workflow_id === desiredWorkflowId)
      ? desiredWorkflowId
      : (entries[0]?.item.workflow_id ?? null);
  const activeEntry = selectWorkflowLibraryEntry({ entries, selectedEntry }, activeWorkflowId) ?? null;
  const activeAccess = getLibraryEntryAccess(activeEntry, authSession, tab);
  // The menu acts on the card it was raised for; once that leaves, it stays closed (the shell clears its anchor).
  const menuEntry = contextMenuTarget
    ? (selectWorkflowLibraryEntry({ entries, selectedEntry }, contextMenuTarget.workflowId) ?? null)
    : null;
  const browseWindow = useMemo(() => ({ startIndex, total, filter }), [startIndex, total, filter]);
  const duplicateGuard = useWorkflowLibraryDuplicateGuard();
  const selectionIntent = useRef(0);
  const selectEntry = useCallback(
    (workflowId: string) => {
      selectionIntent.current += 1;
      onSelect(workflowId);
    },
    [onSelect]
  );
  const handleDuplicated = useCallback(
    (workflowId: string) => {
      // Copies belong to My library; the other places keep their current selection.
      if (tab === 'user') {
        selectEntry(workflowId);
      }
    },
    [selectEntry, tab]
  );

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
        selectEntry(resident.item.workflow_id);
        return;
      }

      // Past the resident window: a new selection or query cancels the intent, even before search debounces.
      const intent = selectionIntent.current;
      const query = workflowUiStore.getSnapshot().libraryQuery;
      void loadWorkflowLibraryRange(target, target).then((ready) => {
        const snapshot = getWorkflowLibraryBrowseSnapshot();
        const loaded = snapshot.entries[target - (snapshot.startIndex ?? 0)];

        if (
          ready &&
          loaded &&
          isCurrent() &&
          selectionIntent.current === intent &&
          snapshot.filter === filter &&
          workflowUiStore.getSnapshot().libraryQuery === query
        ) {
          selectEntry(loaded.item.workflow_id);
        }
      });
    },
    [entries, filter, position, selectEntry, startIndex, total]
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
            isActive={isOpen}
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
            onSelect={selectEntry}
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
            onSelect={selectEntry}
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
          onDuplicated={handleDuplicated}
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
        onDuplicated={handleDuplicated}
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

import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryLayout } from '@features/workflow/ui/WorkflowUiContext';

import { Badge, Box, DataList, HStack, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { useAuthSession } from '@features/identity';
import { getModelBaseLabel } from '@features/models';
import {
  isBundledLibraryWorkflowId,
  savedWorkflowDetailQueryOptions,
  type WorkflowRecordDTO,
} from '@features/workflow/queries';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { MenuActionItem } from '@features/workflow/ui/MenuActionItem';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  useOpenAddModels,
  useWorkflowNotifications,
  useWorkflowProjectSelector,
} from '@features/workflow/ui/WorkflowUiContext';
import { requestWorkflowPublication } from '@features/workflow/ui/workflowUiStore';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { MenuContent } from '@platform/ui/Menu';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useQuery } from '@tanstack/react-query';
import {
  BookmarkIcon,
  CopyIcon,
  EllipsisIcon,
  PencilIcon,
  RefreshCwIcon,
  Trash2Icon,
  WorkflowIcon,
} from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ProjectWorkflowLibraryEntry } from './projectWorkflowEntries';
import type { WorkflowLibraryListTrailingField } from './WorkflowLibraryList';

import { isUpdatableSource, toProjectWorkflowLibraryEntries } from './projectWorkflowEntries';
import { formatTimeSince, useDataTimeSnapshot } from './relativeTime';
import { useOpenLibraryWorkflow } from './useOpenLibraryWorkflow';
import {
  WORKFLOW_DETAIL_WIDE_ONLY_CSS,
  WorkflowDetailRailPlaceholder,
  WorkflowDetailSheet,
  WorkflowMissingNodePacks,
} from './WorkflowDetailSheet';
import { WORKFLOW_LIBRARY_PANES_CSS } from './WorkflowLibraryBrowsePanes';
import {
  getWorkflowLibraryCardId,
  getWorkflowLibraryCardMenuContentId,
  keepCardMenuOpenForRetarget,
  type WorkflowCardMenuAnchor,
  type WorkflowCardContextMenuHandler,
} from './WorkflowLibraryCard';
import { WorkflowDetailFact } from './WorkflowLibraryDetailPanel';
import { WorkflowLibraryGrid } from './WorkflowLibraryGrid';
import { WorkflowLibraryList } from './WorkflowLibraryList';
import { WorkflowThumbnailPlaceholder } from './WorkflowLibraryThumbnail';
import {
  getMissingNodePacks,
  resolveEntryRequirements,
  useModelRequirementDeps,
  WorkflowRequirementsList,
} from './WorkflowRequirementsList';
import { WORKFLOW_THUMBNAIL_ASPECT_RATIO, WorkflowThumbnailImage } from './workflowThumbnail';

/**
 * The project's own workflows, presented through the library's card contract. The dialog filters and orders them
 * (`queryProjectWorkflowEntries`) for its toolbar, tags and result line. Everything here is a local collection
 * command; the library is only consulted for the selected workflow's source, and only on request.
 */

const NO_NODE_PACKS: readonly string[] = [];
/** Keeps the selected rail from re-rendering for template fields it does not show. */
const selectTemplateSummary = (record: WorkflowRecordDTO) => ({ name: record.name, revision: record.revision });

export interface ProjectWorkflowsViewProps {
  contextMenuPoint: WorkflowCardMenuAnchor | null;
  /** What opened the menu last, kept through its close so focus returns there. */
  contextMenuTriggerId: string | null;
  /** The project's workflows the toolbar and tags let through, in the toolbar's order. */
  entries: readonly ProjectWorkflowLibraryEntry[];
  layout: WorkflowLibraryLayout;
  /** Missing-model counts for `entries`, by workflow id. */
  missingCounts: ReadonlyMap<string, number>;
  selectedWorkflowId: string | null;
  /** The date the list's rows show, from the current order. */
  trailingField: WorkflowLibraryListTrailingField;
  onClose: () => void;
  onContextMenu: WorkflowCardContextMenuHandler;
  onContextMenuClose: () => void;
  onPreview: (entry: ProjectWorkflowEntry) => void;
  onSelect: (workflowId: string | null) => void;
}

export const ProjectWorkflowsView = ({
  contextMenuPoint,
  contextMenuTriggerId,
  entries,
  layout,
  missingCounts,
  selectedWorkflowId,
  trailingField,
  onClose,
  onContextMenu,
  onContextMenuClose,
  onPreview,
  onSelect,
}: ProjectWorkflowsViewProps) => {
  const { t } = useTranslation();
  const workflows = useWorkflowProjectSelector((project) => project.workflows);
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const { duplicateWorkflow, removeWorkflow, renameWorkflow, selectWorkflow } = useProjectGraphCommands();
  const [renameTarget, setRenameTarget] = useState<string | null>(null);
  const [removeTarget, setRemoveTarget] = useState<string | null>(null);

  // A filter that hides the selection (and the active workflow) moves the rail to the first match.
  const activeSelectionId = entries.some((entry) => entry.item.workflow_id === selectedWorkflowId)
    ? selectedWorkflowId
    : entries.some((entry) => entry.item.workflow_id === activeWorkflowId)
      ? activeWorkflowId
      : (entries[0]?.item.workflow_id ?? null);
  const selectedEntry = entries.find((entry) => entry.item.workflow_id === activeSelectionId) ?? null;

  const previewWorkflow = useCallback(
    (workflowId: string) => {
      const entry = entries.find((candidate) => candidate.item.workflow_id === workflowId);

      if (entry) {
        onPreview(entry.projectWorkflow);
      }
    },
    [entries, onPreview]
  );
  const openWorkflow = useCallback(
    (workflowId: string) => {
      selectWorkflow(workflowId);
      onClose();
    },
    [onClose, selectWorkflow]
  );
  const handleDuplicate = useCallback(
    (workflowId: string, name: string) => {
      const copyId = duplicateWorkflow(workflowId, t('workflowLibrary.duplicateName', { name }));

      if (copyId) {
        onSelect(copyId);
      }
    },
    [duplicateWorkflow, onSelect, t]
  );
  const confirmRemove = useCallback(() => {
    if (removeTarget) {
      removeWorkflow(removeTarget);
      onSelect(null);
    }
  }, [onSelect, removeTarget, removeWorkflow]);
  const closeRename = useCallback(() => setRenameTarget(null), []);
  const closeRemove = useCallback(() => setRemoveTarget(null), []);
  const submitRename = useCallback(
    (name: string) => {
      if (renameTarget) {
        renameWorkflow(renameTarget, name);
      }
    },
    [renameTarget, renameWorkflow]
  );

  const renameEntry = renameTarget ? workflows.find((entry) => entry.document.id === renameTarget) : undefined;
  const removeEntry = removeTarget ? workflows.find((entry) => entry.document.id === removeTarget) : undefined;

  return (
    <>
      <HStack css={WORKFLOW_LIBRARY_PANES_CSS} align="stretch" flex="1" gap="3" minH="0" minW="0">
        {layout === 'list' ? (
          <WorkflowLibraryList
            activeWorkflowId={activeWorkflowId}
            entries={entries}
            error={null}
            missingCounts={missingCounts}
            openMenuAnchor={contextMenuPoint}
            selectedWorkflowId={activeSelectionId}
            status="loaded"
            onContextMenu={onContextMenu}
            onOpen={openWorkflow}
            onPreview={previewWorkflow}
            onSelect={onSelect}
            trailingField={trailingField}
          />
        ) : (
          <WorkflowLibraryGrid
            activeWorkflowId={activeWorkflowId}
            entries={entries}
            error={null}
            missingCounts={missingCounts}
            openMenuAnchor={contextMenuPoint}
            selectedWorkflowId={activeSelectionId}
            status="loaded"
            onContextMenu={onContextMenu}
            onOpen={openWorkflow}
            onPreview={previewWorkflow}
            onSelect={onSelect}
          />
        )}
        <ProjectWorkflowDetailPanel
          contextMenuPoint={contextMenuPoint}
          contextMenuTriggerId={contextMenuTriggerId}
          entry={selectedEntry ? selectedEntry.projectWorkflow : null}
          isActive={selectedEntry?.item.workflow_id === activeWorkflowId}
          missingCount={selectedEntry ? (missingCounts.get(selectedEntry.item.workflow_id) ?? 0) : 0}
          onContextMenuClose={onContextMenuClose}
          onDuplicate={handleDuplicate}
          onOpen={openWorkflow}
          onPreview={onPreview}
          onRemove={setRemoveTarget}
          onRename={setRenameTarget}
        />
      </HStack>

      <RenameDialog
        initialName={renameEntry?.document.name ?? ''}
        isOpen={renameEntry !== undefined}
        label={t('workflowLibrary.workflowName')}
        submitLabel={t('workflowLibrary.rename')}
        title={t('workflowLibrary.renameTitle')}
        onClose={closeRename}
        onSubmit={submitRename}
      />
      <ConfirmDialog
        body={t('workflowLibrary.removeConfirmBody', {
          name: removeEntry?.document.name || t('workflowLibrary.untitled'),
        })}
        confirmLabel={t('workflowLibrary.remove')}
        isOpen={removeEntry !== undefined}
        title={t('workflowLibrary.removeConfirmTitle')}
        onClose={closeRemove}
        onConfirm={confirmRemove}
      />
    </>
  );
};

interface ProjectWorkflowDetailPanelProps {
  contextMenuPoint: WorkflowCardMenuAnchor | null;
  /** What opened the menu last, kept through its close so focus returns there. */
  contextMenuTriggerId: string | null;
  entry: ProjectWorkflowEntry | null;
  isActive: boolean;
  missingCount: number;
  onContextMenuClose: () => void;
  onDuplicate: (workflowId: string, name: string) => void;
  onOpen: (workflowId: string) => void;
  onPreview: (entry: ProjectWorkflowEntry) => void;
  onRemove: (workflowId: string) => void;
  onRename: (workflowId: string) => void;
}

/**
 * The selected project workflow: preview, name with Open, facts, and the library template it is linked to. The
 * template's current record is read only for the selected workflow, through the saved-workflow detail query the
 * workflow header shares; the query client is account-scoped.
 */
const ProjectWorkflowDetailPanel = ({
  contextMenuPoint,
  contextMenuTriggerId,
  entry,
  isActive,
  missingCount,
  onContextMenuClose,
  onDuplicate,
  onOpen,
  onPreview,
  onRemove,
  onRename,
}: ProjectWorkflowDetailPanelProps) => {
  const panelRef = useRef<HTMLDivElement>(null);
  // Portaled action menus belong to this dialog's visible bounds, rather than the whole viewport.
  const menuPositioning = useMemo(
    () => ({
      boundary: () => panelRef.current?.closest<HTMLElement>('[data-scope="dialog"][data-part="content"]') ?? [],
    }),
    []
  );
  const { t } = useTranslation();
  const session = useAuthSession();
  const deps = useModelRequirementDeps();
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const openAddModels = useOpenAddModels();
  const notify = useWorkflowNotifications();
  const [failedThumbnailUrl, setFailedThumbnailUrl] = useState<string | null>(null);
  const [isUpdateConfirmOpen, setIsUpdateConfirmOpen] = useState(false);
  const dataTime = useDataTimeSnapshot(entry);
  const statusLineId = useId();

  const libraryEntry = useMemo(
    () =>
      entry
        ? toProjectWorkflowLibraryEntries(
            [entry],
            templatesSnapshot.status === 'loaded' ? templatesSnapshot.templates : null
          )[0]!
        : null,
    [entry, templatesSnapshot]
  );
  const resolved = useMemo(
    () =>
      libraryEntry?.enrichment.status === 'ready' ? resolveEntryRequirements(libraryEntry.enrichment, deps) : null,
    [deps, libraryEntry]
  );
  const missingNodePacks =
    libraryEntry?.enrichment.status === 'ready' && templatesSnapshot.status === 'loaded'
      ? getMissingNodePacks(libraryEntry.enrichment, templatesSnapshot.templates)
      : NO_NODE_PACKS;

  const source = entry?.source;
  const isBundledSource = source !== undefined && isBundledLibraryWorkflowId(source.libraryWorkflowId);
  // Bundled templates are rewritten at every server start, so only a user template's revision is comparable.
  const comparesRevision = source !== undefined && !isBundledSource && source.revision !== null;
  /*
   * The template's name and revision come from the full saved-workflow record, workflow JSON included: the server has
   * no metadata-only read by id. Sharing this query (the Call Saved Workflow field and the workflow header use it)
   * costs no second request. It lives in the account-scoped query client with gcTime Infinity, and
   * CallSavedWorkflowSyncRuntime invalidates it when the library cache changes; `select` keeps this rail rendering
   * only for the two fields it shows.
   */
  const template = useQuery({
    ...savedWorkflowDetailQueryOptions(comparesRevision ? source.libraryWorkflowId : ''),
    enabled: comparesRevision,
    select: selectTemplateSummary,
  });
  const templateRevision = comparesRevision ? (template.data?.revision ?? null) : null;
  const isTemplateNewer =
    templateRevision !== null &&
    source?.revision !== null &&
    source !== undefined &&
    templateRevision > source.revision;
  // A template that is gone or no longer readable has nothing to update from.
  const canUpdateFromTemplate = source !== undefined && !(comparesRevision && template.isError);

  const handleReplaced = useCallback(() => notify.success(t('workflowLibrary.updatedFromTemplate')), [notify, t]);
  const { loadPhase, replace } = useOpenLibraryWorkflow(handleReplaced);
  const isReplacing = loadPhase !== 'idle';

  // A pointer point is a fixed rect; the tile button is the menu's trigger, so the menu follows it as it scrolls.
  const contextMenuPositioning = useMemo(
    () =>
      contextMenuPoint?.kind === 'point'
        ? {
            ...menuPositioning,
            getAnchorRect: () => ({ height: 1, width: 1, x: contextMenuPoint.x, y: contextMenuPoint.y }),
            placement: 'bottom-start' as const,
          }
        : { ...menuPositioning, placement: 'bottom-end' as const },
    [contextMenuPoint, menuPositioning]
  );
  const handleContextMenuOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onContextMenuClose();
      }
    },
    [onContextMenuClose]
  );
  // Focus returns to whatever opened the menu: the card's menu button, or the right-clicked card.
  const contextMenuIds = useMemo(
    () =>
      entry
        ? {
            content: getWorkflowLibraryCardMenuContentId(entry.document.id),
            trigger: contextMenuTriggerId ?? getWorkflowLibraryCardId(entry.document.id),
          }
        : undefined,
    [contextMenuTriggerId, entry]
  );
  const moreActionsIds = useTooltipTriggerIds();
  const handleThumbnailError = useCallback(
    () => setFailedThumbnailUrl(libraryEntry?.item.thumbnail_url ?? null),
    [libraryEntry]
  );
  const entryId = entry?.document.id ?? null;
  const entryName = entry?.document.name || t('workflowLibrary.untitled');
  const handleOpen = useCallback(() => entryId && onOpen(entryId), [entryId, onOpen]);
  const handleRename = useCallback(() => entryId && onRename(entryId), [entryId, onRename]);
  const handleDuplicate = useCallback(
    () => entryId && onDuplicate(entryId, entryName),
    [entryId, entryName, onDuplicate]
  );
  const handleRemove = useCallback(() => entryId && onRemove(entryId), [entryId, onRemove]);
  const handlePreview = useCallback(() => entry && onPreview(entry), [entry, onPreview]);
  const handleSaveToLibrary = useCallback(
    () => entryId && requestWorkflowPublication({ kind: 'save-as-new', workflowId: entryId }),
    [entryId]
  );
  const handleUpdateTemplate = useCallback(
    () => entryId && requestWorkflowPublication({ kind: 'update-source', workflowId: entryId }),
    [entryId]
  );
  const openUpdateConfirm = useCallback(() => setIsUpdateConfirmOpen(true), []);
  const closeUpdateConfirm = useCallback(() => setIsUpdateConfirmOpen(false), []);
  // The copy keeps its id and position; the replacement is one undo step.
  const confirmUpdateFromTemplate = useCallback(async () => {
    if (!entryId || !source) {
      return;
    }

    await replace(
      {
        name: template.data?.name || entryName,
        workflow_id: source.libraryWorkflowId,
        ...(templateRevision === null ? {} : { revision: templateRevision }),
      },
      entryId
    );
  }, [entryId, entryName, replace, source, template.data, templateRevision]);

  if (!entry || !libraryEntry) {
    return <WorkflowDetailRailPlaceholder />;
  }

  const workflowId = entry.document.id;
  const name = entryName;
  const thumbnailUrl = libraryEntry.item.thumbnail_url;
  const showThumbnail = Boolean(thumbnailUrl) && thumbnailUrl !== failedThumbnailUrl;
  const lastRun = entry.lastRun ? formatTimeSince(entry.lastRun.completedAt, dataTime) : '';
  const edited = entry.document.updatedAt ? formatTimeSince(entry.document.updatedAt, dataTime) : '';
  const canUpdateSource = isUpdatableSource(source, session);
  const enrichment = libraryEntry.enrichment.status === 'ready' ? libraryEntry.enrichment : null;
  const primaryBase = enrichment?.requirements.primaryBase ?? null;
  const statusLine = isActive
    ? lastRun
      ? t('workflowLibrary.activeRanWhen', { when: lastRun })
      : t('workflowLibrary.activeWorkflowLine')
    : lastRun
      ? t('workflowLibrary.ranWhen', { when: lastRun })
      : '';

  const summaryFacts = [
    primaryBase ? getModelBaseLabel(primaryBase) : null,
    enrichment ? t('workflowLibrary.nodeCount', { count: enrichment.nodeCount }) : null,
    missingNodePacks.length > 0
      ? t('workflowLibrary.nodePacksMissing', { count: missingNodePacks.length })
      : missingCount > 0
        ? t('workflowLibrary.modelsToInstall', { count: missingCount })
        : enrichment
          ? t('workflowLibrary.readyToRun')
          : null,
    isTemplateNewer ? t('workflowLibrary.templateNewer') : null,
    statusLine || null,
  ]
    .filter((fact) => fact !== null)
    .join(' · ');

  const actionItems = (
    <>
      <MenuActionItem
        hint={t('workflowLibrary.openProjectWorkflowHint')}
        icon={WorkflowIcon}
        label={t('workflowLibrary.open')}
        value="open"
        onSelect={handleOpen}
      />
      <MenuActionItem
        hint={t('workflowLibrary.renameProjectWorkflowHint')}
        icon={PencilIcon}
        label={t('workflowLibrary.renameWithEllipsis')}
        value="rename"
        onSelect={handleRename}
      />
      <MenuActionItem
        hint={t('workflowLibrary.duplicateProjectWorkflowHint')}
        icon={CopyIcon}
        label={t('workflowLibrary.duplicate')}
        value="duplicate"
        onSelect={handleDuplicate}
      />
      <MenuActionItem
        hint={t('workflowLibrary.saveToLibraryHint')}
        icon={BookmarkIcon}
        label={t('workflowLibrary.saveToLibraryWithEllipsis')}
        value="save-to-library"
        onSelect={handleSaveToLibrary}
      />
      {canUpdateSource ? (
        <MenuActionItem
          hint={t('workflowLibrary.updateTemplateHint')}
          icon={RefreshCwIcon}
          label={t('workflowLibrary.updateTemplate')}
          value="update-template"
          onSelect={handleUpdateTemplate}
        />
      ) : null}
      <MenuActionItem
        hint={t('workflowLibrary.removeHint')}
        icon={Trash2Icon}
        label={t('workflowLibrary.removeWithEllipsis')}
        tone="danger"
        value="remove"
        onSelect={handleRemove}
      />
    </>
  );

  return (
    <WorkflowDetailSheet
      ref={panelRef}
      actions={actionItems}
      data-project-workflow-detail={workflowId}
      facts={summaryFacts}
      isOpenDisabled={isActive}
      menuPositioning={menuPositioning}
      name={name}
      openLabel={t('workflowLibrary.open')}
      portals={
        <>
          <Menu.Root
            ids={contextMenuIds}
            open={contextMenuPoint !== null}
            positioning={contextMenuPositioning}
            onOpenChange={handleContextMenuOpenChange}
            onPointerDownOutside={keepCardMenuOpenForRetarget}
          >
            <Portal>
              <Menu.Positioner>
                <MenuContent
                  data-workflow-context-menu
                  minW="min(16rem, var(--available-width))"
                  maxW="var(--available-width)"
                >
                  {actionItems}
                </MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>

          <ConfirmDialog
            body={t('workflowLibrary.updateFromTemplateBody', { name })}
            confirmLabel={t('workflowLibrary.updateFromTemplateConfirm')}
            isOpen={isUpdateConfirmOpen}
            title={t('workflowLibrary.updateFromTemplate')}
            onClose={closeUpdateConfirm}
            onConfirm={confirmUpdateFromTemplate}
          />
        </>
      }
      thumbnailSrc={showThumbnail ? thumbnailUrl : null}
      workflowDocument={entry.document}
      workflowId={workflowId}
      onOpen={handleOpen}
    >
      <Scrollable flex="1" label={name} minH="0">
        <Stack gap="2.5" minW="0" p="2.5">
          {/* The expanded sheet's summary is its compact heading: no large preview, no second name. */}
          <Box
            aspectRatio={WORKFLOW_THUMBNAIL_ASPECT_RATIO}
            bg="bg.muted"
            css={WORKFLOW_DETAIL_WIDE_ONLY_CSS}
            overflow="hidden"
            rounded="md"
            w="full"
          >
            {showThumbnail ? (
              <WorkflowThumbnailImage src={thumbnailUrl} onError={handleThumbnailError} />
            ) : (
              <WorkflowThumbnailPlaceholder
                caption={t('workflowLibrary.notRunYet')}
                workflowDocument={entry.document}
              />
            )}
          </Box>

          <Stack css={WORKFLOW_DETAIL_WIDE_ONLY_CSS} gap="0.5" minW="0">
            <Text fontSize="lg" fontWeight="600" minW="0" overflowWrap="anywhere">
              {name}
            </Text>
            {statusLine ? (
              <Text color="fg.muted" fontSize="xs" id={statusLineId}>
                {statusLine}
              </Text>
            ) : null}
          </Stack>

          {primaryBase || enrichment || missingNodePacks.length > 0 ? (
            <HStack flexWrap="wrap" gap="1" minW="0">
              {primaryBase ? <Badge variant="subtle">{getModelBaseLabel(primaryBase)}</Badge> : null}
              {enrichment ? (
                <Badge variant="subtle">{t('workflowLibrary.nodeCount', { count: enrichment.nodeCount })}</Badge>
              ) : null}
              {missingNodePacks.length > 0 ? (
                <Badge bg="bg.error" color="fg.error" data-workflow-missing-node-packs variant="subtle">
                  {t('workflowLibrary.nodePacksMissing', { count: missingNodePacks.length })}
                </Badge>
              ) : null}
            </HStack>
          ) : null}

          {entry.document.description ? (
            <Text color="fg.muted" fontSize="xs" overflowWrap="anywhere">
              {entry.document.description}
            </Text>
          ) : null}

          <Stack
            borderColor="border.subtle"
            borderWidth="1px"
            data-linked-template={source?.libraryWorkflowId ?? 'none'}
            gap="1.5"
            minW="0"
            p="2"
            rounded="md"
          >
            <HStack gap="1.5" justify="space-between" minW="0">
              <Text color="fg.muted" fontSize="xs" fontWeight="600">
                {t('workflowLibrary.linkedTemplate')}
              </Text>
              {isTemplateNewer ? (
                <Badge bg="bg.warning" color="fg.warning" data-template-newer flexShrink={0} variant="subtle">
                  {t('workflowLibrary.templateNewer')}
                </Badge>
              ) : null}
            </HStack>
            <Text color="fg.muted" fontSize="xs" data-workflow-source={source?.libraryWorkflowId ?? 'none'}>
              {source
                ? isBundledSource
                  ? t('workflowLibrary.sourceBundled')
                  : source.revision === null
                    ? t('workflowLibrary.sourceUnknownRevision')
                    : template.data?.name
                      ? t('workflowLibrary.sourceNamedRevision', {
                          name: template.data.name,
                          revision: source.revision,
                        })
                      : t('workflowLibrary.sourceRevision', { revision: source.revision })
                : t('workflowLibrary.sourceNone')}
            </Text>
            {templateRevision !== null ? (
              <Text color="fg.muted" data-template-revision={templateRevision} fontSize="xs">
                {t('workflowLibrary.templateRevision', { revision: templateRevision })}
              </Text>
            ) : null}
            {source ? (
              // Stacked full width: side by side the two labels wrap unevenly in the 18rem rail.
              <Stack gap="1.5" minW="0">
                <Button
                  disabled={!canUpdateFromTemplate || isReplacing}
                  loading={isReplacing}
                  size="xs"
                  variant={isTemplateNewer ? 'solid' : 'outline'}
                  w="full"
                  onClick={openUpdateConfirm}
                >
                  {t('workflowLibrary.updateFromTemplate')}
                </Button>
                {canUpdateSource ? (
                  <Button size="xs" variant="outline" w="full" onClick={handleUpdateTemplate}>
                    {t('workflowLibrary.updateTemplate')}
                  </Button>
                ) : null}
              </Stack>
            ) : null}
          </Stack>

          {libraryEntry.tags.length > 0 ? (
            <HStack flexWrap="wrap" gap="1" minW="0">
              {libraryEntry.tags.map((tag) => (
                <Badge key={tag} variant="outline">
                  {tag}
                </Badge>
              ))}
            </HStack>
          ) : null}

          <WorkflowMissingNodePacks packs={missingNodePacks} />
          <WorkflowRequirementsList errorMessage={null} resolved={resolved} onFindModel={openAddModels} />
          {missingCount > 0 ? (
            <Text color="fg.warning" fontSize="xs">
              {t('workflowLibrary.modelsToInstall', { count: missingCount })}
            </Text>
          ) : null}

          <DataList.Root data-project-workflow-facts gap="1.5" orientation="horizontal">
            <WorkflowDetailFact label={t('workflowLibrary.factLastRun')}>
              {lastRun || t('workflowLibrary.notRunYet')}
            </WorkflowDetailFact>
            {edited ? <WorkflowDetailFact label={t('workflowLibrary.factEdited')}>{edited}</WorkflowDetailFact> : null}
          </DataList.Root>
        </Stack>
      </Scrollable>

      {/* The pinned footer ends in the primary action, as a picker's commit button; the narrow sheet's summary
      carries Open and More instead. */}
      <Stack borderColor="border.subtle" borderTopWidth="1px" gap="2" minW="0" p="2.5">
        <Button
          disabled={libraryEntry.enrichment.status !== 'ready'}
          minW="0"
          size="lg"
          variant="outline"
          w="full"
          onClick={handlePreview}
        >
          <WorkflowIcon />
          {t('workflowLibrary.previewGraph')}
        </Button>
        <HStack css={WORKFLOW_DETAIL_WIDE_ONLY_CSS} gap="2" minW="0">
          {/* The active workflow is already open: the primary slot, disabled, says why through the status line. */}
          <Button
            aria-describedby={isActive ? statusLineId : undefined}
            data-project-workflow-open
            disabled={isActive}
            flex="1"
            minW="0"
            size="lg"
            onClick={handleOpen}
          >
            {t('workflowLibrary.open')}
          </Button>
          <Menu.Root ids={moreActionsIds} positioning={menuPositioning}>
            {/* Inside a dialog the tooltip must sit inside the menu trigger: wrapped the other way round, the
                menu never takes focus and the first pointer move onto it closes it. */}
            <Menu.Trigger asChild>
              <Tooltip content={t('workflowLibrary.moreActions')} ids={moreActionsIds}>
                <IconButton aria-label={t('workflowLibrary.moreActions')} size="lg" variant="outline">
                  <EllipsisIcon />
                </IconButton>
              </Tooltip>
            </Menu.Trigger>
            <Portal>
              <Menu.Positioner>
                <MenuContent minW="min(16rem, var(--available-width))" maxW="var(--available-width)">
                  {actionItems}
                </MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        </HStack>
      </Stack>
    </WorkflowDetailSheet>
  );
};

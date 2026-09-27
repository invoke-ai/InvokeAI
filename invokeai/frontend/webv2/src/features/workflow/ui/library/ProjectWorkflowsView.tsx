import type { ProjectWorkflowEntry } from '@features/workflow/core/types';

import { Badge, Box, Flex, HStack, Icon, Image, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { MenuActionItem } from '@features/workflow/ui/MenuActionItem';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useOpenAddModels, useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { requestWorkflowPublication } from '@features/workflow/ui/workflowUiStore';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { MenuContent } from '@platform/ui/Menu';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import {
  BookmarkIcon,
  CopyIcon,
  EllipsisIcon,
  ImageOffIcon,
  PencilIcon,
  PlusIcon,
  RefreshCwIcon,
  Trash2Icon,
  WorkflowIcon,
} from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  isBundledLibraryWorkflowId,
  isUpdatableSource,
  toProjectWorkflowLibraryEntries,
} from './projectWorkflowEntries';
import { formatRelativeTime } from './relativeTime';
import {
  getWorkflowLibraryCardId,
  getWorkflowLibraryCardMenuContentId,
  keepCardMenuOpenForRetarget,
  type WorkflowCardMenuAnchor,
  type WorkflowLibraryCardProps,
} from './WorkflowLibraryCard';
import { WorkflowLibraryGrid } from './WorkflowLibraryGrid';
import {
  resolveEntryRequirements,
  useModelRequirementDeps,
  useWorkflowLibraryMissingCounts,
  WorkflowRequirementsList,
} from './WorkflowRequirementsList';

/**
 * The project's own workflows, presented through the library's card contract. Everything here is a local
 * collection command; the library is only consulted for the selected workflow's source, and only on request.
 */

const DETAIL_RAIL_WIDTH = '18rem';
const THUMBNAIL_ASPECT_RATIO = 3 / 2;

export interface ProjectWorkflowsViewProps {
  contextMenuPoint: WorkflowCardMenuAnchor | null;
  /** What opened the menu last, kept through its close so focus returns there. */
  contextMenuTriggerId: string | null;
  selectedWorkflowId: string | null;
  onAddWorkflow: () => void;
  onClose: () => void;
  onContextMenu: WorkflowLibraryCardProps['onContextMenu'];
  onContextMenuClose: () => void;
  onPreview: (entry: ProjectWorkflowEntry) => void;
  onSelect: (workflowId: string | null) => void;
}

export const ProjectWorkflowsView = ({
  contextMenuPoint,
  contextMenuTriggerId,
  selectedWorkflowId,
  onAddWorkflow,
  onClose,
  onContextMenu,
  onContextMenuClose,
  onPreview,
  onSelect,
}: ProjectWorkflowsViewProps) => {
  const { t } = useTranslation();
  const workflows = useWorkflowProjectSelector((project) => project.workflows);
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const { createWorkflow, duplicateWorkflow, removeWorkflow, renameWorkflow, selectWorkflow } =
    useProjectGraphCommands();
  const [renameTarget, setRenameTarget] = useState<string | null>(null);
  const [removeTarget, setRemoveTarget] = useState<string | null>(null);

  const entries = useMemo(
    () =>
      toProjectWorkflowLibraryEntries(
        workflows,
        templatesSnapshot.status === 'loaded' ? templatesSnapshot.templates : null
      ),
    [templatesSnapshot, workflows]
  );
  const missingCounts = useWorkflowLibraryMissingCounts(entries);
  const activeSelectionId = entries.some((entry) => entry.item.workflow_id === selectedWorkflowId)
    ? selectedWorkflowId
    : activeWorkflowId;
  const selectedEntry = entries.find((entry) => entry.item.workflow_id === activeSelectionId) ?? null;

  const openWorkflow = useCallback(
    (workflowId: string) => {
      selectWorkflow(workflowId);
      onClose();
    },
    [onClose, selectWorkflow]
  );
  const handleNewWorkflow = useCallback(() => {
    onSelect(createWorkflow());
  }, [createWorkflow, onSelect]);
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
      <Stack flex="1" gap="2" minH="0" minW="0">
        <HStack gap="2" justify="space-between" minW="0">
          <Text color="fg.muted" fontSize="xs" minW="0" truncate>
            {t('workflowLibrary.projectWorkflowCount', { count: workflows.length })}
          </Text>
          <HStack flexShrink={0} gap="1">
            <Button size="xs" variant="outline" onClick={handleNewWorkflow}>
              <PlusIcon />
              {t('workflowLibrary.newWorkflow')}
            </Button>
            <Button size="xs" variant="outline" onClick={onAddWorkflow}>
              <BookmarkIcon />
              {t('workflowLibrary.addWorkflow')}
            </Button>
          </HStack>
        </HStack>
        <HStack align="stretch" flex="1" gap="3" minH="0" minW="0">
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
            onSelect={onSelect}
          />
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
      </Stack>

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
  const { t } = useTranslation();
  const deps = useModelRequirementDeps();
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const openAddModels = useOpenAddModels();
  const [failedThumbnailUrl, setFailedThumbnailUrl] = useState<string | null>(null);

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

  // A pointer point is a fixed rect; the tile button is the menu's trigger, so the menu follows it as it scrolls.
  const contextMenuPositioning = useMemo(
    () =>
      contextMenuPoint?.kind === 'point'
        ? {
            getAnchorRect: () => ({ height: 1, width: 1, x: contextMenuPoint.x, y: contextMenuPoint.y }),
            placement: 'bottom-start' as const,
          }
        : { placement: 'bottom-end' as const },
    [contextMenuPoint]
  );
  const handleContextMenuOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onContextMenuClose();
      }
    },
    [onContextMenuClose]
  );
  // Focus returns to whatever opened the menu: the title's menu button, or the right-clicked card.
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

  if (!entry || !libraryEntry) {
    return (
      <Box borderColor="border.subtle" borderWidth="1px" flexShrink={0} minH="0" rounded="md" w={DETAIL_RAIL_WIDTH} />
    );
  }

  const workflowId = entry.document.id;
  const name = entryName;
  const thumbnailUrl = libraryEntry.item.thumbnail_url;
  const showThumbnail = Boolean(thumbnailUrl) && thumbnailUrl !== failedThumbnailUrl;
  const lastRun = entry.lastRun ? formatRelativeTime(entry.lastRun.completedAt, new Date()) : '';
  const source = entry.source;
  const canUpdateSource = isUpdatableSource(source);

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
    <Stack
      borderColor="border.subtle"
      borderWidth="1px"
      data-project-workflow-detail={workflowId}
      flexShrink={0}
      gap="0"
      minH="0"
      rounded="md"
      w={DETAIL_RAIL_WIDTH}
    >
      <Scrollable flex="1" label={name} minH="0">
        <Stack gap="2" minW="0" p="2.5">
          <Stack gap="1" minW="0">
            <Box aspectRatio={THUMBNAIL_ASPECT_RATIO} bg="bg.muted" overflow="hidden" rounded="md" w="full">
              {showThumbnail ? (
                <Image
                  alt=""
                  h="full"
                  objectFit="cover"
                  src={thumbnailUrl ?? undefined}
                  w="full"
                  onError={handleThumbnailError}
                />
              ) : (
                <Flex align="center" direction="column" gap="1" h="full" justify="center" w="full">
                  <Icon aria-hidden as={ImageOffIcon} boxSize="5" color="fg.subtle" opacity={0.6} />
                  <Text color="fg.subtle" fontSize="2xs">
                    {t('workflowLibrary.notRunYet')}
                  </Text>
                </Flex>
              )}
            </Box>
            {lastRun ? (
              <Text color="fg.subtle" fontSize="2xs">
                {t('workflowLibrary.lastRun', { when: lastRun })}
              </Text>
            ) : null}
          </Stack>

          <HStack gap="1.5" minW="0">
            <Text fontSize="sm" fontWeight="600" minW="0" overflowWrap="anywhere">
              {name}
            </Text>
            {isActive ? (
              <Badge flexShrink={0} size="xs" variant="solid">
                {t('workflowLibrary.activeWorkflow')}
              </Badge>
            ) : null}
          </HStack>

          {entry.document.description ? (
            <Text color="fg.muted" fontSize="2xs" lineClamp={4}>
              {entry.document.description}
            </Text>
          ) : null}

          <Text color="fg.subtle" fontSize="2xs" data-workflow-source={source?.libraryWorkflowId ?? 'none'}>
            {source
              ? isBundledLibraryWorkflowId(source.libraryWorkflowId)
                ? t('workflowLibrary.sourceBundled')
                : source.revision === null
                  ? t('workflowLibrary.sourceUnknownRevision')
                  : t('workflowLibrary.sourceRevision', { revision: source.revision })
              : t('workflowLibrary.sourceNone')}
          </Text>

          {libraryEntry.tags.length > 0 ? (
            <HStack flexWrap="wrap" gap="1" minW="0">
              {libraryEntry.tags.map((tag) => (
                <Badge key={tag} size="xs" variant="subtle">
                  {tag}
                </Badge>
              ))}
            </HStack>
          ) : null}

          <WorkflowRequirementsList errorMessage={null} resolved={resolved} onFindModel={openAddModels} />
        </Stack>
      </Scrollable>

      <Stack borderColor="border.subtle" borderTopWidth="1px" gap="2" p="2.5">
        <HStack gap="2" minW="0">
          <Button disabled={isActive} flex="1" minW="0" size="sm" onClick={handleOpen}>
            {isActive ? t('workflowLibrary.activeWorkflow') : t('workflowLibrary.open')}
          </Button>
          <Menu.Root ids={moreActionsIds}>
            {/* Inside a dialog the tooltip must sit inside the menu trigger: wrapped the other way round, the
                menu never takes focus and the first pointer move onto it closes it. */}
            <Menu.Trigger asChild>
              <Tooltip content={t('workflowLibrary.moreActions')} ids={moreActionsIds}>
                <IconButton aria-label={t('workflowLibrary.moreActions')} size="sm" variant="outline">
                  <EllipsisIcon />
                </IconButton>
              </Tooltip>
            </Menu.Trigger>
            <Portal>
              <Menu.Positioner>
                <MenuContent minW="16rem">{actionItems}</MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        </HStack>
        <Button
          disabled={libraryEntry.enrichment.status !== 'ready'}
          size="sm"
          variant="outline"
          w="full"
          onClick={handlePreview}
        >
          <WorkflowIcon />
          {t('workflowLibrary.previewGraph')}
        </Button>
        {missingCount > 0 ? (
          <Text color="fg.warning" fontSize="2xs">
            {t('workflowLibrary.installModels', { count: missingCount })}
          </Text>
        ) : null}
      </Stack>

      <Menu.Root
        ids={contextMenuIds}
        open={contextMenuPoint !== null}
        positioning={contextMenuPositioning}
        onOpenChange={handleContextMenuOpenChange}
        onPointerDownOutside={keepCardMenuOpenForRetarget}
      >
        <Portal>
          <Menu.Positioner>
            <MenuContent data-workflow-context-menu minW="16rem">
              {actionItems}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Stack>
  );
};

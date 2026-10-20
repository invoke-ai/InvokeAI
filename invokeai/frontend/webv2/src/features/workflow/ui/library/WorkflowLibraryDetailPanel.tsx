import type { StarterInstallSource } from '@features/models';
import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';

import { Badge, Box, HStack, Menu, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import { getStarterModelInstallSources, useInstallActions } from '@features/models';
import {
  createLibraryWorkflow,
  deleteLibraryWorkflow,
  getLibraryWorkflowCached,
  getLibraryWorkflowRecord,
  getLibraryWorkflowRecordCached,
  invalidateWorkflowLibraryCache,
  setLibraryWorkflowPublic,
  updateLibraryWorkflow,
} from '@features/workflow/queries';
import { MenuActionItem } from '@features/workflow/ui/MenuActionItem';
import {
  useOpenAddModels,
  useWorkflowGraphPreview,
  useWorkflowNotifications,
} from '@features/workflow/ui/WorkflowUiContext';
import { parseWorkflowJson } from '@features/workflow/utility';
import { downloadText } from '@platform/browser/downloadBlob';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import {
  Button,
  ConfirmDialog,
  IconButton,
  MenuContent,
  RenameDialog,
  Scrollable,
  Tooltip,
  useTooltipTriggerIds,
} from '@platform/ui';
import { CopyIcon, DownloadIcon, EllipsisIcon, GitForkIcon, PencilIcon, Trash2Icon, WorkflowIcon } from 'lucide-react';
import { useCallback, useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { planLibraryWorkflowOpen } from './useOpenLibraryWorkflow';
import {
  getWorkflowLibraryCardId,
  getWorkflowLibraryCardMenuContentId,
  keepCardMenuOpenForRetarget,
  type WorkflowCardMenuAnchor,
} from './WorkflowLibraryCard';
import { WorkflowLibraryThumbnail } from './WorkflowLibraryThumbnail';
import {
  resolveEntryRequirements,
  useModelRequirementDeps,
  WorkflowRequirementsList,
} from './WorkflowRequirementsList';

/** Install missing models before offering Open; keep the detail rail mounted across selections to prevent flashing. */

const DETAIL_RAIL_WIDTH = '18rem';
const INSTALL_HOVER = { opacity: 0.85 } as const;

export interface WorkflowLibraryDetailPanelProps {
  /** Where a card's right-click asked for the actions menu; null while it is closed. */
  contextMenuPoint: WorkflowCardMenuAnchor | null;
  /** What opened the menu last, kept through its close so focus returns there. */
  contextMenuTriggerId: string | null;
  /** Allows the authenticated owner to change visibility, only on Yours. */
  canChangeVisibility?: boolean;
  entry: WorkflowLibraryEntry | null;
  /** Hides writes to a public template owned by another account. */
  isReadOnly?: boolean;
  /** The shell closes the library when a fork takes the user to a new project. */
  onClose: () => void;
  onContextMenuClose: () => void;
  onDeleted: () => void;
  /** Carries the copy's id so the shell can select it once the list refreshes. */
  onDuplicated: (workflowId: string) => void;
  /** Adds the first copy, or asks what to do when the project already holds one. */
  onOpen: (item: WorkflowLibraryListItem) => void;
  onPreview: (entry: WorkflowLibraryEntry) => void;
  /** The project's workflows, so the rail can say when opening will ask about existing copies. */
  projectWorkflows: readonly ProjectWorkflowEntry[];
}

const toFileSlug = (name: string): string => name.trim().replaceAll(/\s+/g, '-').toLowerCase() || 'workflow';

export const WorkflowLibraryDetailPanel = ({
  canChangeVisibility = false,
  contextMenuPoint,
  contextMenuTriggerId,
  entry,
  isReadOnly = false,
  onClose,
  onContextMenuClose,
  onDeleted,
  onDuplicated,
  onOpen,
  onPreview,
  projectWorkflows,
}: WorkflowLibraryDetailPanelProps) => {
  const { t } = useTranslation();
  const deps = useModelRequirementDeps();
  const notify = useWorkflowNotifications();
  const { openDocumentInNewProject } = useWorkflowGraphPreview();
  const openAddModels = useOpenAddModels();
  const { installMany } = useInstallActions();
  // The name is captured on request: a delete moves the selection, or empties this panel, before the dialog closes.
  const [deleteTargetName, setDeleteTargetName] = useState<string | null>(null);
  // Guard duplicate creation until the copy exists, including copies landing outside the visible category.
  const [isDuplicatePending, setIsDuplicatePending] = useState(false);
  const isDuplicatePendingRef = useRef(false);
  const [visibilityRequest, setVisibilityRequest] = useState<
    { workflowId: string; status: 'pending' } | { workflowId: string; message: string; status: 'error' } | null
  >(null);
  const [visibilityOverride, setVisibilityOverride] = useState<{
    workflowId: string;
    isPublic: boolean;
    previousItem: WorkflowLibraryListItem;
  } | null>(null);
  const visibilityInputRef = useRef<HTMLInputElement>(null);
  const restoreVisibilityFocus = useRef(false);

  const enrichment = entry?.enrichment ?? null;
  // Shares the grid's per-entry cache, so selecting a card the badges already
  // resolved costs nothing.
  const resolved = useMemo(
    () => (enrichment?.status === 'ready' ? resolveEntryRequirements(enrichment, deps) : null),
    [deps, enrichment]
  );
  const installableCount = resolved?.filter((requirement) => requirement.status === 'installable').length ?? 0;

  const hasCopies = useMemo(
    () => (entry ? planLibraryWorkflowOpen(projectWorkflows, entry.item.workflow_id).kind === 'choose' : false),
    [entry, projectWorkflows]
  );
  const handleOpen = useCallback(() => {
    if (entry) {
      onOpen(entry.item);
    }
  }, [entry, onOpen]);

  const handlePreview = useCallback(() => {
    if (entry && entry.enrichment.status === 'ready') {
      onPreview(entry);
    }
  }, [entry, onPreview]);

  // Close the library before navigating to Add Models.
  const handleFindModel = useCallback(
    (query: string) => {
      openAddModels(query);
      onClose();
    },
    [onClose, openAddModels]
  );

  const install = useCallback(async () => {
    const owner = captureAccountScope();
    const starterBySource = new Map(deps.starterModels.map((starter) => [starter.source, starter]));
    const requests: StarterInstallSource[] = [];
    const seen = new Set<string>();

    for (const requirement of resolved ?? []) {
      const starter = requirement.starterMatch ? starterBySource.get(requirement.starterMatch.source) : undefined;

      if (requirement.status !== 'installable' || !starter) {
        continue;
      }

      for (const source of getStarterModelInstallSources(starter)) {
        // Two requirements routinely share a dependency (an encoder, a VAE),
        // and an install may have started between this render and the click.
        if (seen.has(source.source) || deps.activeInstallSources.has(source.source)) {
          continue;
        }

        seen.add(source.source);
        requests.push(source);
      }
    }

    if (requests.length === 0) {
      return;
    }

    const queued = await installMany(requests);

    // One notice for the whole set; `installMany` toasts its own failures.
    if (queued > 0 && isAccountScopeCurrent(owner)) {
      notify.success(t('workflowLibrary.installQueued'));
    }
  }, [deps, installMany, notify, resolved, t]);
  const handleInstall = useCallback(() => void install(), [install]);

  /** Duplicate the library record directly without loading or modifying the active project graph. */
  const duplicate = useCallback(async () => {
    if (!entry || isDuplicatePendingRef.current) {
      return;
    }

    const owner = captureAccountScope();
    const { name } = entry.item;

    isDuplicatePendingRef.current = true;
    setIsDuplicatePending(true);

    try {
      const raw = await getLibraryWorkflowCached(entry.item.workflow_id, {
        expectedRevision: entry.item.revision,
        signal: owner.signal,
      });

      assertAccountScopeCurrent(owner);

      const { id: _id, ...copy } = raw;
      const meta = typeof copy.meta === 'object' && copy.meta !== null ? (copy.meta as Record<string, unknown>) : {};
      const workflowId = await createLibraryWorkflow(
        {
          ...copy,
          // A copy of a default is still the user's own workflow.
          meta: { ...meta, category: 'user' },
          name: t('workflowLibrary.duplicateName', { name: name || t('workflowLibrary.untitled') }),
        },
        owner.signal
      );

      assertAccountScopeCurrent(owner);
      invalidateWorkflowLibraryCache();
      // The copy is forced into the user category, so from the Browse tab it
      // lands out of sight — the notice is the only thing that says it worked.
      notify.success(t('workflowLibrary.duplicated'));
      onDuplicated(workflowId);
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.duplicateFailed'), getApiErrorMessage(error, t('common.unknownError')));
      }
    } finally {
      if (isAccountScopeCurrent(owner)) {
        isDuplicatePendingRef.current = false;
        setIsDuplicatePending(false);
      }
    }
  }, [entry, notify, onDuplicated, t]);
  const handleDuplicate = useCallback(() => void duplicate(), [duplicate]);

  const [isRenameOpen, setIsRenameOpen] = useState(false);
  const openRename = useCallback(() => setIsRenameOpen(true), []);
  const closeRename = useCallback(() => setIsRenameOpen(false), []);
  // A rename is a content write at the record's current revision: the live record, never a cached one, is the base.
  const submitRename = useCallback(
    async (nextName: string) => {
      if (!entry || isReadOnly) {
        return;
      }

      const owner = captureAccountScope();

      try {
        const record = await getLibraryWorkflowRecord(entry.item.workflow_id, owner.signal);

        assertAccountScopeCurrent(owner);
        await updateLibraryWorkflow(
          entry.item.workflow_id,
          { ...record.workflow, name: nextName },
          { expectedRevision: record.revision, signal: owner.signal }
        );
        assertAccountScopeCurrent(owner);
        invalidateWorkflowLibraryCache(entry.item.workflow_id);
        notify.success(t('workflowLibrary.renamed'));
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        notify.error(t('workflowLibrary.renameFailed'), getApiErrorMessage(error, t('common.unknownError')));
        // Rejecting keeps the dialog, and the typed name, open for another try.
        throw error;
      }
    },
    [entry, isReadOnly, notify, t]
  );

  const fork = useCallback(async () => {
    if (!entry) {
      return;
    }

    const owner = captureAccountScope();

    try {
      const record = await getLibraryWorkflowRecordCached(entry.item.workflow_id, {
        expectedRevision: entry.item.revision,
        signal: owner.signal,
      });

      assertAccountScopeCurrent(owner);

      const { document } = parseWorkflowJson({ ...record.workflow, id: record.workflow_id });

      // The port creates and activates a fresh project first, so the project
      // the library was opened from is left exactly as it was.
      openDocumentInNewProject(document, entry.item.name, {
        libraryWorkflowId: record.workflow_id,
        ...(record.category === 'user' && record.user_id ? { ownerUserId: record.user_id } : {}),
        revision: record.revision,
      });
      onClose();
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.loadFailed'), getApiErrorMessage(error, t('common.unknownError')));
      }
    }
  }, [entry, notify, onClose, openDocumentInNewProject, t]);
  const handleFork = useCallback(() => void fork(), [fork]);

  const download = useCallback(async () => {
    if (!entry) {
      return;
    }

    const owner = captureAccountScope();

    try {
      const raw = await getLibraryWorkflowCached(entry.item.workflow_id, {
        expectedRevision: entry.item.revision,
        signal: owner.signal,
      });

      assertAccountScopeCurrent(owner);
      downloadText(JSON.stringify(raw, null, 2), `${toFileSlug(entry.item.name)}.json`, 'application/json');
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.loadFailed'), getApiErrorMessage(error, t('common.unknownError')));
      }
    }
  }, [entry, notify, t]);
  const handleDownload = useCallback(() => void download(), [download]);

  const openDeleteConfirm = useCallback(() => {
    if (entry && !isReadOnly) {
      setDeleteTargetName(entry.item.name || t('workflowLibrary.untitled'));
    }
  }, [entry, isReadOnly, t]);
  const closeDeleteConfirm = useCallback(() => setDeleteTargetName(null), []);

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
            content: getWorkflowLibraryCardMenuContentId(entry.item.workflow_id),
            trigger: contextMenuTriggerId ?? getWorkflowLibraryCardId(entry.item.workflow_id),
          }
        : undefined,
    [contextMenuTriggerId, entry]
  );
  const moreActionsIds = useTooltipTriggerIds();

  const confirmDelete = useCallback(async () => {
    if (!entry || isReadOnly) {
      return;
    }

    const owner = captureAccountScope();

    try {
      await deleteLibraryWorkflow(entry.item.workflow_id, owner.signal);

      assertAccountScopeCurrent(owner);
      invalidateWorkflowLibraryCache();
      onDeleted();
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.deleteFailed'), getApiErrorMessage(error, t('common.unknownError')));
      }
    }
  }, [entry, isReadOnly, notify, onDeleted, t]);

  const item = entry?.item ?? null;
  const tags = entry?.tags ?? [];
  const name = item ? item.name || t('workflowLibrary.untitled') : '';
  const itemIsPublic =
    item && visibilityOverride?.workflowId === item.workflow_id && visibilityOverride.previousItem === item
      ? visibilityOverride.isPublic
      : item?.is_public === true;
  const isVisibilityPending =
    item !== null && visibilityRequest?.workflowId === item.workflow_id && visibilityRequest.status === 'pending';
  const visibilityError =
    item !== null && visibilityRequest?.workflowId === item.workflow_id && visibilityRequest.status === 'error'
      ? visibilityRequest.message
      : null;
  useLayoutEffect(() => {
    if (restoreVisibilityFocus.current && !isVisibilityPending) {
      restoreVisibilityFocus.current = false;
      visibilityInputRef.current?.focus();
    }
  }, [isVisibilityPending, itemIsPublic]);
  const updateVisibility = useCallback(
    async (isPublic: boolean) => {
      if (!item || !canChangeVisibility || isVisibilityPending || isPublic === itemIsPublic) {
        return;
      }

      const owner = captureAccountScope();
      const workflowId = item.workflow_id;
      const previousItem = item;
      restoreVisibilityFocus.current = document.activeElement === visibilityInputRef.current;
      setVisibilityRequest({ status: 'pending', workflowId });

      try {
        const record = await setLibraryWorkflowPublic(workflowId, isPublic, owner.signal);

        assertAccountScopeCurrent(owner);
        setVisibilityOverride({ isPublic: record.is_public === true, previousItem, workflowId });
        invalidateWorkflowLibraryCache(workflowId);
        setVisibilityRequest(null);
      } catch (error) {
        if (isAccountScopeCurrent(owner)) {
          setVisibilityRequest({
            message: getApiErrorMessage(error, t('common.unknownError')),
            status: 'error',
            workflowId,
          });
        }
      } finally {
        setVisibilityRequest((current) =>
          current?.workflowId === workflowId && current.status === 'pending' ? null : current
        );
      }
    },
    [canChangeVisibility, isVisibilityPending, item, itemIsPublic, t]
  );
  const handleVisibilityClick = useCallback(
    (event: MouseEvent<HTMLInputElement>) => {
      event.preventDefault();
      void updateVisibility(!itemIsPublic);
    },
    [itemIsPublic, updateVisibility]
  );
  const openLabel = hasCopies ? t('workflowLibrary.openWithEllipsis') : t('workflowLibrary.open');

  // One item set behind both the rail's overflow button and a card's right-click.
  const actionItems = (
    <>
      <MenuActionItem
        hint={hasCopies ? t('workflowLibrary.openExistingCopiesHint') : t('workflowLibrary.openHint')}
        icon={WorkflowIcon}
        label={openLabel}
        value="open"
        onSelect={handleOpen}
      />
      <MenuActionItem
        hint={t('workflowLibrary.duplicateHint')}
        icon={CopyIcon}
        isDisabled={isDuplicatePending}
        label={t('workflowLibrary.duplicate')}
        value="duplicate"
        onSelect={handleDuplicate}
      />
      <MenuActionItem
        hint={t('workflowLibrary.forkIntoProjectHint')}
        icon={GitForkIcon}
        label={t('workflowLibrary.forkIntoProject')}
        value="fork-into-project"
        onSelect={handleFork}
      />
      <MenuActionItem
        hint={t('workflowLibrary.downloadJsonHint')}
        icon={DownloadIcon}
        label={t('workflowLibrary.downloadJson')}
        value="download-json"
        onSelect={handleDownload}
      />
      {item?.category === 'user' && !isReadOnly ? (
        <MenuActionItem
          hint={t('workflowLibrary.renameTemplateHint')}
          icon={PencilIcon}
          label={t('workflowLibrary.renameWithEllipsis')}
          value="rename"
          onSelect={openRename}
        />
      ) : null}
      {item?.category === 'user' && !isReadOnly ? (
        // Bundled defaults are not the account's to delete.
        <MenuActionItem
          hint={t('workflowLibrary.deleteHint')}
          icon={Trash2Icon}
          label={t('workflowLibrary.delete')}
          tone="danger"
          value="delete"
          onSelect={openDeleteConfirm}
        />
      ) : null}
    </>
  );

  return (
    <>
      {item ? (
        <Stack
          borderColor="border.subtle"
          borderWidth="1px"
          data-workflow-detail={item.workflow_id}
          flexShrink={0}
          gap="0"
          minH="0"
          rounded="md"
          w={DETAIL_RAIL_WIDTH}
        >
          <Scrollable flex="1" label={name} minH="0">
            <Stack gap="2" minW="0" p="2.5">
              <WorkflowLibraryThumbnail
                key={item.workflow_id}
                canEdit={item.category === 'user' && !isReadOnly}
                item={item}
                workflowDocument={enrichment?.status === 'ready' ? enrichment.document : null}
              />

              {/*
               * Wrap full names, including delimiter-free strings, in the detail rail; zero content min-width permits
               * containment without truncation.
               */}
              <Text fontSize="lg" fontWeight="600" minW="0" overflowWrap="anywhere">
                {name}
              </Text>

              {item.category === 'user' ? (
                <Stack gap="1">
                  <HStack align="center" justify="space-between" minW="0">
                    <Text color="fg.muted" fontSize="xs">
                      {t('workflowLibrary.visibility')}
                    </Text>
                    <Badge data-workflow-visibility={itemIsPublic ? 'public' : 'private'} variant="subtle">
                      {t(itemIsPublic ? 'workflowLibrary.public' : 'workflowLibrary.private')}
                    </Badge>
                  </HStack>
                  {canChangeVisibility ? (
                    <Stack gap="1">
                      <Switch.Root checked={itemIsPublic} colorPalette="accent" disabled={isVisibilityPending} readOnly>
                        <Switch.HiddenInput
                          key={itemIsPublic ? 'public' : 'private'}
                          onClick={handleVisibilityClick}
                          ref={visibilityInputRef}
                        />
                        <Switch.Control>
                          <Switch.Thumb />
                        </Switch.Control>
                        <Switch.Label srOnly>{t('workflowLibrary.publicWorkflow')}</Switch.Label>
                      </Switch.Root>
                      <Text color="fg.subtle" fontSize="xs">
                        {t('workflowLibrary.publicVisibilityHelp')}
                      </Text>
                    </Stack>
                  ) : null}
                  {visibilityError ? (
                    <Text color="fg.error" fontSize="xs" role="alert">
                      {t('workflowLibrary.visibilityUpdateFailed')}: {visibilityError}
                    </Text>
                  ) : null}
                </Stack>
              ) : null}

              {item.description ? (
                <Text color="fg.muted" fontSize="xs" lineClamp={4}>
                  {item.description}
                </Text>
              ) : null}

              {tags.length > 0 ? (
                <HStack flexWrap="wrap" gap="1" minW="0">
                  {tags.map((tag) => (
                    <Badge key={tag} variant="subtle">
                      {tag}
                    </Badge>
                  ))}
                </HStack>
              ) : null}

              <WorkflowRequirementsList
                errorMessage={enrichment?.status === 'error' ? enrichment.message : null}
                resolved={resolved}
                onFindModel={handleFindModel}
              />
            </Stack>
          </Scrollable>

          <Stack borderColor="border.subtle" borderTopWidth="1px" gap="2" p="2.5">
            <HStack gap="2" minW="0">
              {installableCount > 0 ? (
                <Button
                  bg="bg.warning"
                  color="fg.warning"
                  flex="1"
                  minW="0"
                  size="lg"
                  _hover={INSTALL_HOVER}
                  onClick={handleInstall}
                >
                  {t('workflowLibrary.installModels', { count: installableCount })}
                </Button>
              ) : (
                <Button flex="1" minW="0" size="lg" onClick={handleOpen}>
                  {openLabel}
                </Button>
              )}
              <Menu.Root ids={moreActionsIds}>
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
                    <MenuContent minW="16rem">{actionItems}</MenuContent>
                  </Menu.Positioner>
                </Portal>
              </Menu.Root>
            </HStack>
            <Button
              disabled={enrichment?.status !== 'ready'}
              size="lg"
              variant="outline"
              w="full"
              onClick={handlePreview}
            >
              <WorkflowIcon />
              {t('workflowLibrary.previewGraph')}
            </Button>
          </Stack>

          {/* Naming the card as the trigger makes this a nested layer of the dialog: the
          dialog's focus trap then lets the menu keep focus, and closing returns it to the card. */}
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

          <RenameDialog
            initialName={name}
            isOpen={isRenameOpen && !isReadOnly}
            label={t('workflowLibrary.templateName')}
            submitLabel={t('workflowLibrary.rename')}
            title={t('workflowLibrary.renameTemplateTitle')}
            onClose={closeRename}
            onSubmit={submitRename}
          />
        </Stack>
      ) : (
        <Box borderColor="border.subtle" borderWidth="1px" flexShrink={0} minH="0" rounded="md" w={DETAIL_RAIL_WIDTH} />
      )}
      {/* Outside the rail: a delete empties or retargets it while this dialog animates out. */}
      <ConfirmDialog
        body={t('workflowLibrary.deleteConfirmBody', { name: deleteTargetName ?? '' })}
        confirmLabel={t('workflowLibrary.delete')}
        isOpen={deleteTargetName !== null && !isReadOnly}
        title={t('workflowLibrary.deleteConfirmTitle')}
        onClose={closeDeleteConfirm}
        onConfirm={confirmDelete}
      />
    </>
  );
};

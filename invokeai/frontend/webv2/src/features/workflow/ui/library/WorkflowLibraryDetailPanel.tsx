import type { StarterInstallSource } from '@features/models';
import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';

import { Badge, Box, Flex, HStack, Icon, Image, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { getStarterModelInstallSources, useInstallActions } from '@features/models';
import {
  createLibraryWorkflow,
  deleteLibraryWorkflow,
  getLibraryWorkflowCached,
  getLibraryWorkflowRecordCached,
  invalidateWorkflowLibraryCache,
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
  Scrollable,
  Tooltip,
  useTooltipTriggerIds,
} from '@platform/ui';
import {
  CopyIcon,
  CopyPlusIcon,
  DownloadIcon,
  EllipsisIcon,
  GitForkIcon,
  ImageOffIcon,
  Trash2Icon,
  WorkflowIcon,
} from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { OpenLibraryWorkflowMode } from './useOpenLibraryWorkflow';

import { formatRelativeTime } from './relativeTime';
import { planLibraryWorkflowOpen } from './useOpenLibraryWorkflow';
import { getWorkflowLibraryCardId } from './WorkflowLibraryCard';
import {
  resolveEntryRequirements,
  useModelRequirementDeps,
  WorkflowRequirementsList,
} from './WorkflowRequirementsList';

/** Install missing models before offering Open; keep the detail rail mounted across selections to prevent flashing. */

const DETAIL_RAIL_WIDTH = '18rem';
const THUMBNAIL_ASPECT_RATIO = 3 / 2;
const INSTALL_HOVER = { opacity: 0.85 } as const;

export interface WorkflowLibraryDetailPanelProps {
  /** Where a card's right-click asked for the actions menu; null while it is closed. */
  contextMenuPoint: { x: number; y: number } | null;
  entry: WorkflowLibraryEntry | null;
  /** The shell closes the library when a fork takes the user to a new project. */
  onClose: () => void;
  onContextMenuClose: () => void;
  onDeleted: () => void;
  /** Carries the copy's id so the shell can select it once the list refreshes. */
  onDuplicated: (workflowId: string) => void;
  onOpen: (item: WorkflowLibraryListItem, mode: OpenLibraryWorkflowMode) => void;
  onPreview: (entry: WorkflowLibraryEntry) => void;
  /** Activates a project copy the project already made from this template. */
  onResume: (workflowId: string) => void;
  /** The project's workflows, so the rail can offer the existing copies of this template. */
  projectWorkflows: readonly ProjectWorkflowEntry[];
}

/** A right-click on another card moves the menu there (its own handler re-anchors it) rather than dismissing it. */
const keepOpenForCardRightClick = (event: { detail: { originalEvent: Event }; preventDefault(): void }) => {
  const original = event.detail.originalEvent;

  if (original instanceof PointerEvent && original.button === 2 && isCardTarget(original.target)) {
    event.preventDefault();
  }
};

const isCardTarget = (target: EventTarget | null): boolean =>
  target instanceof Element && target.closest('[data-workflow-card]') !== null;

const toFileSlug = (name: string): string => name.trim().replaceAll(/\s+/g, '-').toLowerCase() || 'workflow';

const CHOOSER_POSITIONING = { placement: 'top-start' } as const;

const ProjectCopyItem = ({
  copy,
  onResume,
}: {
  copy: ProjectWorkflowEntry;
  onResume: (workflowId: string) => void;
}) => {
  const { t } = useTranslation();
  const handleSelect = useCallback(() => onResume(copy.document.id), [copy.document.id, onResume]);

  return (
    <MenuActionItem
      icon={WorkflowIcon}
      label={copy.document.name || t('workflowLibrary.untitled')}
      value={`resume:${copy.document.id}`}
      onSelect={handleSelect}
    />
  );
};

export const WorkflowLibraryDetailPanel = ({
  contextMenuPoint,
  entry,
  onClose,
  onContextMenuClose,
  onDeleted,
  onDuplicated,
  onOpen,
  onPreview,
  onResume,
  projectWorkflows,
}: WorkflowLibraryDetailPanelProps) => {
  const { t } = useTranslation();
  const deps = useModelRequirementDeps();
  const notify = useWorkflowNotifications();
  const { openDocumentInNewProject } = useWorkflowGraphPreview();
  const openAddModels = useOpenAddModels();
  const { installMany } = useInstallActions();
  // Keyed by URL rather than a boolean, so a selection change re-arms the
  // thumbnail without an effect resetting the flag.
  const [failedThumbnailUrl, setFailedThumbnailUrl] = useState<string | null>(null);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  // Guard duplicate creation until the copy exists, including copies landing outside the visible category.
  const [isDuplicatePending, setIsDuplicatePending] = useState(false);
  const isDuplicatePendingRef = useRef(false);

  const enrichment = entry?.enrichment ?? null;
  // Shares the grid's per-entry cache, so selecting a card the badges already
  // resolved costs nothing.
  const resolved = useMemo(
    () => (enrichment?.status === 'ready' ? resolveEntryRequirements(enrichment, deps) : null),
    [deps, enrichment]
  );
  const installableCount = resolved?.filter((requirement) => requirement.status === 'installable').length ?? 0;

  const openPlan = useMemo(
    () => (entry ? planLibraryWorkflowOpen(projectWorkflows, entry.item.workflow_id) : null),
    [entry, projectWorkflows]
  );
  const [isCopyChooserOpen, setIsCopyChooserOpen] = useState(false);

  /** Open lands in the project: the first copy is added, an existing one resumed, several offered as a choice. */
  const handleOpen = useCallback(() => {
    if (!entry || !openPlan) {
      return;
    }

    if (openPlan.kind === 'choose') {
      setIsCopyChooserOpen(true);
      return;
    }

    if (openPlan.kind === 'resume') {
      onResume(openPlan.workflowId);
      return;
    }

    onOpen(entry.item, 'resume-or-add');
  }, [entry, onOpen, onResume, openPlan]);
  const handleAddCopy = useCallback(() => {
    if (entry) {
      onOpen(entry.item, 'add-copy');
    }
  }, [entry, onOpen]);
  const handleCopyChooserOpenChange = useCallback((event: { open: boolean }) => setIsCopyChooserOpen(event.open), []);

  const handlePreview = useCallback(() => {
    if (entry && entry.enrichment.status === 'ready') {
      onPreview(entry);
    }
  }, [entry, onPreview]);

  const handleThumbnailError = useCallback(() => setFailedThumbnailUrl(entry?.item.thumbnail_url ?? null), [entry]);

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
      const raw = await getLibraryWorkflowCached(entry.item.workflow_id, owner.signal);

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

  const fork = useCallback(async () => {
    if (!entry) {
      return;
    }

    const owner = captureAccountScope();

    try {
      const record = await getLibraryWorkflowRecordCached(entry.item.workflow_id, owner.signal);

      assertAccountScopeCurrent(owner);

      const { document } = parseWorkflowJson({ ...record.workflow, id: record.workflow_id });

      // The port creates and activates a fresh project first, so the project
      // the library was opened from is left exactly as it was.
      openDocumentInNewProject(document, entry.item.name, {
        libraryWorkflowId: record.workflow_id,
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
      const raw = await getLibraryWorkflowCached(entry.item.workflow_id, owner.signal);

      assertAccountScopeCurrent(owner);
      downloadText(JSON.stringify(raw, null, 2), `${toFileSlug(entry.item.name)}.json`, 'application/json');
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.loadFailed'), getApiErrorMessage(error, t('common.unknownError')));
      }
    }
  }, [entry, notify, t]);
  const handleDownload = useCallback(() => void download(), [download]);

  const openDeleteConfirm = useCallback(() => setIsDeleteConfirmOpen(true), []);
  const closeDeleteConfirm = useCallback(() => setIsDeleteConfirmOpen(false), []);

  const contextMenuPositioning = useMemo(
    () => ({
      getAnchorRect: () =>
        contextMenuPoint ? { height: 1, width: 1, x: contextMenuPoint.x, y: contextMenuPoint.y } : null,
      placement: 'bottom-start' as const,
    }),
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
  const contextMenuIds = useMemo(
    () => (entry ? { trigger: getWorkflowLibraryCardId(entry.item.workflow_id) } : undefined),
    [entry]
  );
  // The chooser is anchored to the button that opened it, so closing returns focus there.
  const openButtonId = entry ? `${getWorkflowLibraryCardId(entry.item.workflow_id)}-open` : undefined;
  const chooserIds = useMemo(() => (openButtonId ? { trigger: openButtonId } : undefined), [openButtonId]);
  const moreActionsIds = useTooltipTriggerIds();

  const confirmDelete = useCallback(async () => {
    if (!entry) {
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
  }, [entry, notify, onDeleted, t]);

  if (!entry) {
    return (
      <Box borderColor="border.subtle" borderWidth="1px" flexShrink={0} minH="0" rounded="md" w={DETAIL_RAIL_WIDTH} />
    );
  }

  const { item, tags } = entry;
  const name = item.name || t('workflowLibrary.untitled');
  const showThumbnail = Boolean(item.thumbnail_url) && item.thumbnail_url !== failedThumbnailUrl;
  const lastRun = item.last_run_at ? formatRelativeTime(item.last_run_at, new Date()) : '';
  const caption = lastRun
    ? t('workflowLibrary.lastRun', { when: lastRun })
    : showThumbnail
      ? t('workflowLibrary.sampleOutput')
      : null;

  const hasCopies = openPlan !== null && openPlan.kind !== 'add';
  const openLabel = hasCopies ? t('workflowLibrary.openProjectCopy') : t('workflowLibrary.open');

  // One item set behind both the rail's overflow button and a card's right-click.
  const actionItems = (
    <>
      <MenuActionItem
        hint={hasCopies ? t('workflowLibrary.openProjectCopyHint') : t('workflowLibrary.openHint')}
        icon={WorkflowIcon}
        label={openLabel}
        value="open"
        onSelect={handleOpen}
      />
      {hasCopies ? (
        <MenuActionItem
          hint={t('workflowLibrary.addAnotherCopyHint')}
          icon={CopyPlusIcon}
          label={t('workflowLibrary.addAnotherCopy')}
          value="add-copy"
          onSelect={handleAddCopy}
        />
      ) : null}
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
      {item.category === 'user' ? (
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
          <Stack gap="1" minW="0">
            <Box aspectRatio={THUMBNAIL_ASPECT_RATIO} bg="bg.muted" overflow="hidden" rounded="md" w="full">
              {showThumbnail ? (
                <Image
                  alt=""
                  h="full"
                  objectFit="cover"
                  src={item.thumbnail_url ?? undefined}
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
            {caption ? (
              <Text color="fg.subtle" fontSize="2xs">
                {caption}
              </Text>
            ) : null}
          </Stack>

          {/*
           * Wrap full names, including delimiter-free strings, in the detail rail; zero content min-width permits
           * containment without truncation.
           */}
          <Text fontSize="sm" fontWeight="600" minW="0" overflowWrap="anywhere">
            {name}
          </Text>

          {item.description ? (
            <Text color="fg.muted" fontSize="2xs" lineClamp={4}>
              {item.description}
            </Text>
          ) : null}

          {tags.length > 0 ? (
            <HStack flexWrap="wrap" gap="1" minW="0">
              {tags.map((tag) => (
                <Badge key={tag} size="xs" variant="subtle">
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
              size="sm"
              _hover={INSTALL_HOVER}
              onClick={handleInstall}
            >
              {t('workflowLibrary.installModels', { count: installableCount })}
            </Button>
          ) : (
            <Button flex="1" id={openButtonId} minW="0" size="sm" onClick={handleOpen}>
              {openLabel}
            </Button>
          )}
          <Menu.Root ids={moreActionsIds}>
            <Tooltip content={t('workflowLibrary.moreActions')} ids={moreActionsIds}>
              <Menu.Trigger asChild>
                <IconButton aria-label={t('workflowLibrary.moreActions')} size="sm" variant="outline">
                  <EllipsisIcon />
                </IconButton>
              </Menu.Trigger>
            </Tooltip>
            <Portal>
              <Menu.Positioner>
                <MenuContent minW="16rem">{actionItems}</MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        </HStack>
        <Button disabled={enrichment?.status !== 'ready'} size="sm" variant="outline" w="full" onClick={handlePreview}>
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
        onPointerDownOutside={keepOpenForCardRightClick}
      >
        <Portal>
          <Menu.Positioner>
            <MenuContent data-workflow-context-menu minW="16rem">
              {actionItems}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      {/* Several copies exist: the chooser lists them by name; another copy is a separate action. */}
      <Menu.Root
        ids={chooserIds}
        open={isCopyChooserOpen && openPlan?.kind === 'choose'}
        positioning={CHOOSER_POSITIONING}
        onOpenChange={handleCopyChooserOpenChange}
      >
        <Portal>
          <Menu.Positioner>
            <MenuContent data-workflow-copy-chooser minW="16rem">
              <Menu.ItemGroup>
                <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  {t('workflowLibrary.chooseProjectCopy')}
                </Menu.ItemGroupLabel>
                {openPlan?.kind === 'choose'
                  ? openPlan.copies.map((copy) => (
                      <ProjectCopyItem key={copy.document.id} copy={copy} onResume={onResume} />
                    ))
                  : null}
              </Menu.ItemGroup>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>

      <ConfirmDialog
        body={t('workflowLibrary.deleteConfirmBody', { name })}
        confirmLabel={t('workflowLibrary.delete')}
        isOpen={isDeleteConfirmOpen}
        title={t('workflowLibrary.deleteConfirmTitle')}
        onClose={closeDeleteConfirm}
        onConfirm={confirmDelete}
      />
    </Stack>
  );
};

import type { StarterInstallSource } from '@features/models';
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';

import { Badge, Box, Flex, HStack, Icon, Image, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { getStarterModelInstallSources, useInstallActions } from '@features/models';
import { resolveWorkflowModelRequirements } from '@features/workflow/core/modelRequirements';
import {
  createLibraryWorkflow,
  deleteLibraryWorkflow,
  getLibraryWorkflowCached,
  invalidateWorkflowLibraryCache,
} from '@features/workflow/queries';
import { useWorkflowGraphPreview, useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import { parseWorkflowJson } from '@features/workflow/utility';
import { downloadText } from '@platform/browser/downloadBlob';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, ConfirmDialog, IconButton, MenuContent, Scrollable } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import {
  CopyIcon,
  DownloadIcon,
  EllipsisIcon,
  GitForkIcon,
  ImageOffIcon,
  Trash2Icon,
  WorkflowIcon,
} from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { formatRelativeTime } from './relativeTime';
import { useModelRequirementDeps, WorkflowRequirementsList } from './WorkflowRequirementsList';

/**
 * The library's right rail: everything about the selected workflow that
 * decides whether to open it — sample output, description, tags, and the
 * models it needs — plus the one action that follows from that answer. When
 * models are missing the primary button installs them instead of opening a
 * workflow that would fail on its first run; everything rarer (duplicate,
 * fork, download, delete) lives behind the overflow menu.
 *
 * The panel updates in place across selection changes: no keys, no remounts,
 * so switching cards never flashes the rail.
 */

const DETAIL_RAIL_WIDTH = '18rem';
const THUMBNAIL_ASPECT_RATIO = 3 / 2;
const INSTALL_HOVER = { opacity: 0.85 } as const;
const MENU_ITEM_LAYOUT = { alignItems: 'center', gap: '2.5', py: '1' } as const;
const MENU_ITEM_ICON_PROPS = { boxSize: '3.5', flexShrink: 0 } as const;

export interface WorkflowLibraryDetailPanelProps {
  entry: WorkflowLibraryEntry | null;
  /** The shell closes the library when a fork takes the user to a new project. */
  onClose: () => void;
  onDeleted: () => void;
  /** Carries the copy's id so the shell can select it once the list refreshes. */
  onDuplicated: (workflowId: string) => void;
  onOpen: (item: WorkflowLibraryListItem) => void;
  onPreview: (entry: WorkflowLibraryEntry) => void;
}

const toFileSlug = (name: string): string => name.trim().replaceAll(/\s+/g, '-').toLowerCase() || 'workflow';

export const WorkflowLibraryDetailPanel = ({
  entry,
  onClose,
  onDeleted,
  onDuplicated,
  onOpen,
  onPreview,
}: WorkflowLibraryDetailPanelProps) => {
  const { t } = useTranslation();
  const deps = useModelRequirementDeps();
  const notify = useWorkflowNotifications();
  const { openDocumentInNewProject } = useWorkflowGraphPreview();
  const { installMany } = useInstallActions();
  // Keyed by URL rather than a boolean, so a selection change re-arms the
  // thumbnail without an effect resetting the flag.
  const [failedThumbnailUrl, setFailedThumbnailUrl] = useState<string | null>(null);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);

  const enrichment = entry?.enrichment ?? null;
  const resolved = useMemo(
    () =>
      enrichment?.status === 'ready'
        ? resolveWorkflowModelRequirements(enrichment.requirements.requirements, deps)
        : null,
    [deps, enrichment]
  );
  const installableCount = resolved?.filter((requirement) => requirement.status === 'installable').length ?? 0;

  const handleOpen = useCallback(() => {
    if (entry) {
      onOpen(entry.item);
    }
  }, [entry, onOpen]);

  const handlePreview = useCallback(() => {
    if (entry) {
      onPreview(entry);
    }
  }, [entry, onPreview]);

  const handleThumbnailError = useCallback(() => setFailedThumbnailUrl(entry?.item.thumbnail_url ?? null), [entry]);

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

  /**
   * Copies the library *record* — not the project graph, and never the
   * original — so a bundled default can be adapted without the editor ever
   * loading it.
   */
  const duplicate = useCallback(async () => {
    if (!entry) {
      return;
    }

    const owner = captureAccountScope();
    const { name } = entry.item;

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
      onDuplicated(workflowId);
    } catch (error) {
      if (isAccountScopeCurrent(owner)) {
        notify.error(t('workflowLibrary.duplicateFailed'), getApiErrorMessage(error, t('common.unknownError')));
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
      const raw = await getLibraryWorkflowCached(entry.item.workflow_id, owner.signal);

      assertAccountScopeCurrent(owner);

      const { document } = parseWorkflowJson(raw);

      // The port creates and activates a fresh project first, so the project
      // the library was opened from is left exactly as it was.
      openDocumentInNewProject(document, entry.item.name);
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

  const confirmDelete = useCallback(async () => {
    if (!entry) {
      return;
    }

    const owner = captureAccountScope();

    try {
      await deleteLibraryWorkflow(entry.item.workflow_id, owner.signal);

      assertAccountScopeCurrent(owner);
      // The invalidation refetches the visible pages; the shell only has to
      // let go of the selection this row held.
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

          <MiddleTruncate fontSize="sm" fontWeight="600" minW="0" text={name} />

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
          />
        </Stack>
      </Scrollable>

      <Stack borderColor="border.subtle" borderTopWidth="1px" gap="2" p="2.5">
        <HStack gap="2" minW="0">
          {installableCount > 0 ? (
            // The theme's amber, with the ramp's darkest step for text — that
            // step is dark in both light and dark schemes, so the one warning
            // token carries the whole treatment.
            <Button
              bg="fg.warning"
              color="neutral.950"
              flex="1"
              minW="0"
              size="sm"
              _hover={INSTALL_HOVER}
              onClick={handleInstall}
            >
              {t('workflowLibrary.installModels', { count: installableCount })}
            </Button>
          ) : (
            <Button flex="1" minW="0" size="sm" onClick={handleOpen}>
              {t('workflowLibrary.open')}
            </Button>
          )}
          <Menu.Root>
            <Menu.Trigger asChild>
              <IconButton aria-label={t('workflowLibrary.moreActions')} size="sm" variant="outline">
                <EllipsisIcon />
              </IconButton>
            </Menu.Trigger>
            <Portal>
              <Menu.Positioner>
                <MenuContent minW="12rem">
                  <Menu.Item {...MENU_ITEM_LAYOUT} data-menu-item="open" value="open" onClick={handleOpen}>
                    <Icon as={WorkflowIcon} {...MENU_ITEM_ICON_PROPS} />
                    <Menu.ItemText>{t('workflowLibrary.open')}</Menu.ItemText>
                  </Menu.Item>
                  <Menu.Item
                    {...MENU_ITEM_LAYOUT}
                    data-menu-item="duplicate"
                    value="duplicate"
                    onClick={handleDuplicate}
                  >
                    <Icon as={CopyIcon} {...MENU_ITEM_ICON_PROPS} />
                    <Menu.ItemText>{t('workflowLibrary.duplicate')}</Menu.ItemText>
                  </Menu.Item>
                  <Menu.Item
                    {...MENU_ITEM_LAYOUT}
                    data-menu-item="fork-into-project"
                    value="fork-into-project"
                    onClick={handleFork}
                  >
                    <Icon as={GitForkIcon} {...MENU_ITEM_ICON_PROPS} />
                    <Menu.ItemText>{t('workflowLibrary.forkIntoProject')}</Menu.ItemText>
                  </Menu.Item>
                  <Menu.Item
                    {...MENU_ITEM_LAYOUT}
                    data-menu-item="download-json"
                    value="download-json"
                    onClick={handleDownload}
                  >
                    <Icon as={DownloadIcon} {...MENU_ITEM_ICON_PROPS} />
                    <Menu.ItemText>{t('workflowLibrary.downloadJson')}</Menu.ItemText>
                  </Menu.Item>
                  {item.category === 'user' ? (
                    // Bundled defaults are not the account's to delete.
                    <Menu.Item
                      {...MENU_ITEM_LAYOUT}
                      color="fg.error"
                      data-menu-item="delete"
                      value="delete"
                      onClick={openDeleteConfirm}
                    >
                      <Icon as={Trash2Icon} {...MENU_ITEM_ICON_PROPS} />
                      <Menu.ItemText>{t('workflowLibrary.delete')}</Menu.ItemText>
                    </Menu.Item>
                  ) : null}
                </MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        </HStack>
        <Button size="sm" variant="outline" w="full" onClick={handlePreview}>
          <WorkflowIcon />
          {t('workflowLibrary.previewGraph')}
        </Button>
      </Stack>

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

import { Box, Flex, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { Link } from '@tanstack/react-router';
import { IconButton, ConfirmDialog, MenuContent, RenameDialog, toaster } from '@workbench/components/ui';
import {
  deleteLibraryProject,
  duplicateLibraryProject,
  renameLibraryProject,
  type ProjectSummary,
} from '@workbench/projects/library';
import { exportLibraryProject } from '@workbench/projects/projectFile';
import {
  ArrowRightIcon,
  CopyIcon,
  EllipsisVerticalIcon,
  FileDownIcon,
  FolderIcon,
  PencilIcon,
  Trash2Icon,
} from 'lucide-react';
import { useCallback, useMemo, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { formatRelativeTime } from './formatRelativeTime';

const CARD_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const LINK_STYLE = { inset: 0, position: 'absolute' } as const;
const MENU_ITEM_DELETE_HOVER = { bg: 'bg.error', color: 'fg.error' } as const;
const MENU_POSITION_BOTTOM_END = { placement: 'bottom-end' } as const;

/**
 * One saved project in the Home grid. The whole card is a deep link into the
 * editor (`/app?project=…` — hovering preloads the editor chunk); the corner
 * menu carries the library actions, which all run against the server without
 * mounting the editor.
 */
export const ProjectCard = ({ summary }: { summary: ProjectSummary }) => {
  const { t } = useTranslation();
  const [isRenameOpen, setIsRenameOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isActionsOpen, setIsActionsOpen] = useState(false);
  const [contextMenuTarget, setContextMenuTarget] = useState<{ x: number; y: number } | null>(null);

  const projectSearch = useMemo(() => ({ project: summary.id }), [summary.id]);
  const menuPositioning = useMemo(
    () =>
      contextMenuTarget
        ? {
            getAnchorRect: () => ({ height: 1, width: 1, x: contextMenuTarget.x, y: contextMenuTarget.y }),
            placement: 'bottom-start' as const,
          }
        : MENU_POSITION_BOTTOM_END,
    [contextMenuTarget]
  );

  const handleRename = useCallback(
    async (name: string) => {
      try {
        await renameLibraryProject(summary.id, name);
      } catch (error) {
        toaster.create({
          description: error instanceof Error ? error.message : undefined,
          title: t('projects.renameFailed'),
          type: 'error',
        });
        throw error;
      }
    },
    [summary.id, t]
  );

  const handleDuplicate = useCallback(async () => {
    try {
      const copy = await duplicateLibraryProject(summary.id);

      toaster.create({
        description: t('projects.projectDuplicatedDescription', { name: copy.name }),
        title: t('projects.projectDuplicated'),
        type: 'success',
      });
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: t('projects.duplicateFailed'),
        type: 'error',
      });
    }
  }, [summary.id, t]);

  const handleExport = useCallback(async () => {
    try {
      await exportLibraryProject(summary.id);
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: t('projects.exportFailed'),
        type: 'error',
      });
    }
  }, [summary.id, t]);

  const handleDelete = useCallback(async () => {
    try {
      await deleteLibraryProject(summary.id);
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: t('projects.deleteFailed'),
        type: 'error',
      });
    }
  }, [summary.id, t]);
  const handleContextMenu = useCallback((event: MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    setContextMenuTarget({ x: event.clientX, y: event.clientY });
    setIsActionsOpen(true);
  }, []);
  const handleOpenChange = useCallback((event: { open: boolean }) => {
    setIsActionsOpen(event.open);

    if (!event.open) {
      setContextMenuTarget(null);
    }
  }, []);
  const clearContextMenuTarget = useCallback(() => setContextMenuTarget(null), []);
  const openRenameDialog = useCallback(() => setIsRenameOpen(true), []);
  const closeRenameDialog = useCallback(() => setIsRenameOpen(false), []);
  const openDeleteDialog = useCallback(() => setIsDeleteOpen(true), []);
  const closeDeleteDialog = useCallback(() => setIsDeleteOpen(false), []);

  return (
    <Box
      bg="bg.subtle"
      borderColor="border.subtle"
      borderWidth="1px"
      overflow="hidden"
      position="relative"
      rounded="lg"
      transition="border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease"
      _hover={CARD_HOVER}
      onContextMenu={handleContextMenu}
    >
      <Link
        aria-label={t('projects.openProjectLabel', { name: summary.name })}
        search={projectSearch}
        style={LINK_STYLE}
        to="/app"
      />
      <Flex align="center" bg="bg.muted" h="24" justify="center" pointerEvents="none">
        <Icon as={FolderIcon} boxSize="8" color="fg.subtle" opacity={0.6} />
      </Flex>
      <Flex align="center" gap="2" p="3" pointerEvents="none">
        <Stack flex="1" gap="0" minW="0">
          <Text fontSize="xs" fontWeight="600" truncate>
            {summary.name}
          </Text>
          <Text color="fg.muted" fontSize="2xs">
            {t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
          </Text>
        </Stack>
      </Flex>
      <Box bottom="2" pointerEvents="auto" position="absolute" right="2" zIndex="1">
        <Menu.Root open={isActionsOpen} positioning={menuPositioning} onOpenChange={handleOpenChange}>
          <Menu.Trigger asChild>
            <IconButton
              aria-label={t('common.actions')}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={clearContextMenuTarget}
            >
              <EllipsisVerticalIcon />
            </IconButton>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="44">
                <Menu.Item asChild value="open">
                  <Link search={projectSearch} to="/app">
                    <Icon as={ArrowRightIcon} boxSize="3.5" />
                    {t('common.open')}
                  </Link>
                </Menu.Item>
                <Menu.Item value="rename" onClick={openRenameDialog}>
                  <Icon as={PencilIcon} boxSize="3.5" />
                  {t('projects.renameWithEllipsis')}
                </Menu.Item>
                <Menu.Item value="duplicate" onClick={handleDuplicate}>
                  <Icon as={CopyIcon} boxSize="3.5" />
                  {t('common.duplicate')}
                </Menu.Item>
                <Menu.Item value="export" onClick={handleExport}>
                  <Icon as={FileDownIcon} boxSize="3.5" />
                  {t('common.export')}
                </Menu.Item>
                <Menu.Separator />
                <Menu.Item color="fg.error" value="delete" _hover={MENU_ITEM_DELETE_HOVER} onClick={openDeleteDialog}>
                  <Icon as={Trash2Icon} boxSize="3.5" />
                  {t('common.delete')}…
                </Menu.Item>
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Box>

      <RenameDialog
        initialName={summary.name}
        isOpen={isRenameOpen}
        onClose={closeRenameDialog}
        onSubmit={handleRename}
      />

      <ConfirmDialog
        body={t('projects.deleteProjectCardBody', { name: summary.name })}
        confirmLabel={t('projects.deleteProject')}
        isOpen={isDeleteOpen}
        title={t('projects.deleteProjectQuestion')}
        onClose={closeDeleteDialog}
        onConfirm={handleDelete}
      />
    </Box>
  );
};

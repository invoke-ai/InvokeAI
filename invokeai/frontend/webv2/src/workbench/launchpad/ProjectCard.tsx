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
import { useState } from 'react';

import { formatRelativeTime } from './formatRelativeTime';

/**
 * One saved project in the Home grid. The whole card is a deep link into the
 * editor (`/app?project=…` — hovering preloads the editor chunk); the corner
 * menu carries the library actions, which all run against the server without
 * mounting the editor.
 */
export const ProjectCard = ({ summary }: { summary: ProjectSummary }) => {
  const [isRenameOpen, setIsRenameOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isActionsOpen, setIsActionsOpen] = useState(false);
  const [contextMenuTarget, setContextMenuTarget] = useState<{ x: number; y: number } | null>(null);

  const handleRename = async (name: string) => {
    try {
      await renameLibraryProject(summary.id, name);
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: 'Rename failed',
        type: 'error',
      });
      throw error;
    }
  };

  const handleDuplicate = async () => {
    try {
      const copy = await duplicateLibraryProject(summary.id);

      toaster.create({ description: `"${copy.name}" was created.`, title: 'Project duplicated', type: 'success' });
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: 'Duplicate failed',
        type: 'error',
      });
    }
  };

  const handleExport = async () => {
    try {
      await exportLibraryProject(summary.id);
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: 'Export failed',
        type: 'error',
      });
    }
  };

  const handleDelete = async () => {
    try {
      await deleteLibraryProject(summary.id);
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: 'Delete failed',
        type: 'error',
      });
    }
  };

  return (
    <Box
      bg="bg.subtle"
      borderColor="border.subtle"
      borderWidth="1px"
      overflow="hidden"
      position="relative"
      rounded="lg"
      transition="border-color 0.15s ease, background 0.15s ease"
      _hover={{ bg: 'bg.muted', borderColor: 'border.emphasized' }}
      onContextMenu={(event) => {
        event.preventDefault();
        setContextMenuTarget({ x: event.clientX, y: event.clientY });
        setIsActionsOpen(true);
      }}
    >
      <Link
        aria-label={`Open ${summary.name}`}
        search={{ project: summary.id }}
        style={{ inset: 0, position: 'absolute' }}
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
            Edited {formatRelativeTime(summary.updatedAt)}
          </Text>
        </Stack>
      </Flex>
      <Box bottom="2" pointerEvents="auto" position="absolute" right="2" zIndex="1">
        <Menu.Root
          open={isActionsOpen}
          positioning={
            contextMenuTarget
              ? {
                  getAnchorRect: () => ({ height: 1, width: 1, x: contextMenuTarget.x, y: contextMenuTarget.y }),
                  placement: 'bottom-start',
                }
              : { placement: 'bottom-end' }
          }
          onOpenChange={(event) => {
            setIsActionsOpen(event.open);

            if (!event.open) {
              setContextMenuTarget(null);
            }
          }}
        >
          <Menu.Trigger asChild>
            <IconButton
              aria-label={`Actions for ${summary.name}`}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={() => setContextMenuTarget(null)}
            >
              <EllipsisVerticalIcon />
            </IconButton>
          </Menu.Trigger>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="44">
                <Menu.Item asChild value="open">
                  <Link search={{ project: summary.id }} to="/app">
                    <Icon as={ArrowRightIcon} boxSize="3.5" />
                    Open
                  </Link>
                </Menu.Item>
                <Menu.Item value="rename" onClick={() => setIsRenameOpen(true)}>
                  <Icon as={PencilIcon} boxSize="3.5" />
                  Rename…
                </Menu.Item>
                <Menu.Item value="duplicate" onClick={() => void handleDuplicate()}>
                  <Icon as={CopyIcon} boxSize="3.5" />
                  Duplicate
                </Menu.Item>
                <Menu.Item value="export" onClick={() => void handleExport()}>
                  <Icon as={FileDownIcon} boxSize="3.5" />
                  Export
                </Menu.Item>
                <Menu.Separator />
                <Menu.Item
                  color="fg.error"
                  value="delete"
                  _hover={{ bg: 'bg.error', color: 'fg.error' }}
                  onClick={() => setIsDeleteOpen(true)}
                >
                  <Icon as={Trash2Icon} boxSize="3.5" />
                  Delete…
                </Menu.Item>
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Box>

      <RenameDialog
        initialName={summary.name}
        isOpen={isRenameOpen}
        onClose={() => setIsRenameOpen(false)}
        onSubmit={handleRename}
      />

      <ConfirmDialog
        body={`Delete "${summary.name}"? The project is removed from the server permanently.`}
        confirmLabel="Delete project"
        isOpen={isDeleteOpen}
        title="Delete project?"
        onClose={() => setIsDeleteOpen(false)}
        onConfirm={handleDelete}
      />
    </Box>
  );
};

import { Checkbox, Dialog, Flex, HStack, Icon, Menu, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { BrushCleaningIcon, FolderSearchIcon, MoreHorizontalIcon, RefreshCcwIcon } from 'lucide-react';
import { useEffect, useState } from 'react';

import { Button, CloseButton, IconButton } from '../../components/ui/Button';
import { MenuContent } from '../../components/ui/Menu';
import { deleteOrphanedModels, emptyModelCache, getOrphanedModels } from '../../models/api';
import { refreshModels } from '../../models/modelsStore';
import { formatBytes } from '../../models/taxonomy';
import type { OrphanedModelInfo } from '../../models/types';
import { useNotify } from '../../useNotify';

/**
 * Library maintenance: refresh, clean up orphaned model folders (files on disk
 * with no database record), and empty the in-memory model cache.
 */
export const MaintenanceMenu = () => {
  const notify = useNotify();
  const [isSyncDialogOpen, setIsSyncDialogOpen] = useState(false);

  const handleEmptyCache = async () => {
    try {
      await emptyModelCache();
      notify.success('Model cache emptied');
    } catch (error) {
      notify.error('Failed to empty model cache', error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <>
      <Menu.Root positioning={{ placement: 'bottom-end' }}>
        <Menu.Trigger asChild>
          <IconButton aria-label="Model library maintenance" size="xs" variant="ghost">
            <Icon as={MoreHorizontalIcon} boxSize="4" />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="14rem">
              <Menu.Item value="refresh" onClick={() => void refreshModels()}>
                <Icon as={RefreshCcwIcon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">Refresh model list</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="sync" onClick={() => setIsSyncDialogOpen(true)}>
                <Icon as={FolderSearchIcon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">Clean up orphaned models…</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="empty-cache" onClick={() => void handleEmptyCache()}>
                <Icon as={BrushCleaningIcon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">Empty model cache</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      {isSyncDialogOpen ? <OrphanedModelsDialog onClose={() => setIsSyncDialogOpen(false)} /> : null}
    </>
  );
};

const OrphanedModelsDialog = ({ onClose }: { onClose: () => void }) => {
  const notify = useNotify();
  const [orphans, setOrphans] = useState<OrphanedModelInfo[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Nothing pre-selected: deleting files on disk must be an explicit choice.
  const [selectedPaths, setSelectedPaths] = useState<ReadonlySet<string>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    let isStale = false;

    getOrphanedModels()
      .then((result) => {
        if (!isStale) {
          setOrphans(result);
        }
      })
      .catch((error: unknown) => {
        if (!isStale) {
          setLoadError(error instanceof Error ? error.message : 'Failed to scan for orphaned models.');
        }
      });

    return () => {
      isStale = true;
    };
  }, []);

  const togglePath = (path: string) => {
    setSelectedPaths((current) => {
      const next = new Set(current);

      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }

      return next;
    });
  };

  const handleDelete = async () => {
    setIsDeleting(true);

    try {
      const result = await deleteOrphanedModels([...selectedPaths]);
      const errorCount = Object.keys(result.errors).length;

      if (errorCount > 0) {
        notify.error(
          'Orphaned model cleanup',
          `${result.deleted.length} deleted, ${errorCount} failed: ${Object.values(result.errors)[0]}`
        );
      } else {
        notify.success('Orphaned model cleanup', `${result.deleted.length} orphaned model folder(s) deleted.`);
      }

      void refreshModels();
      onClose();
    } catch (error) {
      notify.error('Orphaned model cleanup failed', error instanceof Error ? error.message : String(error));
    } finally {
      setIsDeleting(false);
    }
  };

  const totalSelectedBytes =
    orphans?.filter((orphan) => selectedPaths.has(orphan.path)).reduce((sum, orphan) => sum + orphan.size_bytes, 0) ??
    0;

  return (
    <Dialog.Root
      open
      placement="center"
      scrollBehavior="inside"
      size="md"
      onOpenChange={(event) => {
        if (!event.open) {
          onClose();
        }
      }}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content bg="bg.surface" borderColor="border.subtle" borderWidth="1px" color="fg.default">
            <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
              <Stack gap="0.5">
                <Dialog.Title fontSize="sm" fontWeight="700">
                  Orphaned Models
                </Dialog.Title>
                <Text color="fg.subtle" fontSize="2xs">
                  Folders in the models directory with no database record — usually leftovers from failed installs or
                  external deletes.
                </Text>
              </Stack>
            </Dialog.Header>
            <Dialog.Body>
              {loadError ? (
                <Text color="red.400" fontSize="xs" py="4">
                  {loadError}
                </Text>
              ) : orphans === null ? (
                <Flex align="center" justify="center" py="8">
                  <Spinner color="fg.subtle" size="sm" />
                </Flex>
              ) : orphans.length === 0 ? (
                <Text color="fg.muted" fontSize="xs" py="4" textAlign="center">
                  No orphaned model folders found — your library is clean.
                </Text>
              ) : (
                <Stack gap="1.5" py="2">
                  <Checkbox.Root
                    checked={
                      selectedPaths.size === 0 ? false : selectedPaths.size === orphans.length ? true : 'indeterminate'
                    }
                    colorPalette="theme"
                    ps="2"
                    size="sm"
                    onCheckedChange={() => {
                      setSelectedPaths(
                        selectedPaths.size === orphans.length
                          ? new Set()
                          : new Set(orphans.map((orphan) => orphan.path))
                      );
                    }}
                  >
                    <Checkbox.HiddenInput />
                    <Checkbox.Control />
                    <Checkbox.Label color="fg.muted" fontSize="2xs">
                      Select all ({orphans.length})
                    </Checkbox.Label>
                  </Checkbox.Root>
                  {orphans.map((orphan) => (
                    <HStack
                      key={orphan.path}
                      bg="bg.panel"
                      borderColor="border.subtle"
                      borderWidth="1px"
                      gap="2"
                      p="2"
                      rounded="md"
                    >
                      <Checkbox.Root
                        checked={selectedPaths.has(orphan.path)}
                        colorPalette="theme"
                        size="sm"
                        onCheckedChange={() => togglePath(orphan.path)}
                      >
                        <Checkbox.HiddenInput />
                        <Checkbox.Control />
                      </Checkbox.Root>
                      <Stack flex="1" gap="0" minW="0">
                        <Text fontSize="2xs" fontWeight="600" overflowWrap="anywhere">
                          {orphan.path}
                        </Text>
                        <Text color="fg.subtle" fontSize="2xs">
                          {orphan.files.length} file{orphan.files.length === 1 ? '' : 's'} ·{' '}
                          {formatBytes(orphan.size_bytes)}
                        </Text>
                      </Stack>
                    </HStack>
                  ))}
                </Stack>
              )}
            </Dialog.Body>
            <Dialog.Footer gap="2">
              <Button disabled={isDeleting} size="xs" variant="ghost" onClick={onClose}>
                Close
              </Button>
              <Button
                colorPalette="red"
                disabled={selectedPaths.size === 0 || !orphans || orphans.length === 0}
                loading={isDeleting}
                size="xs"
                variant="solid"
                onClick={() => void handleDelete()}
              >
                Delete Selected ({formatBytes(totalSelectedBytes)})
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton color="fg.muted" size="sm" />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

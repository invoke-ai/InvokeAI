import { chakra, Dialog, Icon, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import { ArrowRightIcon, FileUpIcon } from 'lucide-react';

import { Button, CloseButton } from './ui/Button';
import { Scrollable } from './ui/Scrollable';
import { formatRelativeTime } from '../home/formatRelativeTime';
import { useNotify } from '../useNotify';
import { useWorkbench } from '../WorkbenchContext';
import { refreshProjectLibrary, useProjectLibrary, type ProjectSummary } from '../projects/library';
import { importProjectFile, pickProjectFile } from '../projects/projectFile';
import { adoptProjectRecord, hydrateProjectFromServer } from '../projects/syncedPersistence';

/**
 * "Open project…" from the tab bar: the saved projects that are not already
 * open as tabs, plus import. Selecting one hydrates its document from the
 * library and opens it in place — no navigation, the editor stays mounted.
 */
export const OpenProjectDialog = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const { state, dispatch } = useWorkbench();
  const notify = useNotify();
  const library = useProjectLibrary();
  const [busyProjectId, setBusyProjectId] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      void refreshProjectLibrary();
    }
  }, [isOpen]);

  const openProjectIds = new Set(state.projects.map((project) => project.id));
  const available = library.summaries.filter((summary) => !openProjectIds.has(summary.id));

  const openProject = async (summary: ProjectSummary) => {
    setBusyProjectId(summary.id);

    const project = await hydrateProjectFromServer(summary.id);

    setBusyProjectId(null);

    if (!project) {
      notify.error('Could not open project', `"${summary.name}" could not be loaded from the server.`);
      void refreshProjectLibrary();

      return;
    }

    dispatch({ project, type: 'openProject' });
    onClose();
  };

  const handleImport = async () => {
    const file = await pickProjectFile();

    if (!file) {
      return;
    }

    try {
      const record = await importProjectFile(file);
      const project = adoptProjectRecord(record);

      if (project) {
        dispatch({ project, type: 'openProject' });
        onClose();
      }
    } catch (error) {
      notify.error('Import failed', error instanceof Error ? error.message : undefined);
    }
  };

  return (
    <Dialog.Root
      lazyMount
      open={isOpen}
      placement="center"
      size="sm"
      unmountOnExit
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
            <Dialog.Header>
              <Dialog.Title fontSize="sm" fontWeight="700">
                Open project
              </Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Scrollable maxH="72">
                <Stack gap="1">
                  {available.map((summary) => (
                    <chakra.button
                      key={summary.id}
                      alignItems="center"
                      disabled={busyProjectId !== null}
                      display="flex"
                      gap="2.5"
                      px="2.5"
                      py="2"
                      rounded="md"
                      textAlign="start"
                      type="button"
                      w="full"
                      _disabled={{ opacity: 0.6 }}
                      _hover={{ bg: 'bg.surfaceRaised' }}
                      onClick={() => void openProject(summary)}
                    >
                      <Stack flex="1" gap="0" minW="0">
                        <Text fontSize="xs" fontWeight="600" truncate>
                          {summary.name}
                        </Text>
                        <Text color="fg.muted" fontSize="2xs">
                          Edited {formatRelativeTime(summary.updatedAt)}
                        </Text>
                      </Stack>
                      {busyProjectId === summary.id ? (
                        <Spinner color="fg.muted" size="xs" />
                      ) : (
                        <Icon as={ArrowRightIcon} boxSize="3.5" color="fg.muted" />
                      )}
                    </chakra.button>
                  ))}
                  {available.length === 0 ? (
                    <Text color="fg.muted" fontSize="xs" px="2.5" py="4" textAlign="center">
                      {library.summaries.length === 0
                        ? 'No saved projects yet.'
                        : 'All saved projects are already open.'}
                    </Text>
                  ) : null}
                </Stack>
              </Scrollable>
            </Dialog.Body>
            <Dialog.Footer gap="2" justifyContent="space-between">
              <Button size="xs" variant="outline" onClick={() => void handleImport()}>
                <FileUpIcon />
                Import…
              </Button>
              <Button size="xs" variant="ghost" onClick={onClose}>
                Cancel
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

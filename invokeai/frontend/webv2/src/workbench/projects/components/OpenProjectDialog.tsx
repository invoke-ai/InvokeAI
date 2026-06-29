import { Dialog, Icon, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { Button, CloseButton, Row, Scrollable } from '@workbench/components/ui';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { refreshProjectLibrary, useProjectLibrarySelector, type ProjectSummary } from '@workbench/projects/library';
import { importProjectFile, pickProjectFile } from '@workbench/projects/projectFile';
import { adoptProjectRecord, hydrateProjectFromServer } from '@workbench/projects/syncedPersistence';
import { useNotify } from '@workbench/useNotify';
import { flushGenerateDrafts } from '@workbench/widgets/generate/generateDraftRegistry';
import { useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { areArraysEqual } from '@workbench/workbenchSelectors';
import { ArrowRightIcon, FileUpIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

const disabledRowStyles = { opacity: 0.6 } as const;

/**
 * "Open project…" from the tab bar: the saved projects that are not already
 * open as tabs, plus import. Selecting one hydrates its document from the
 * library and opens it in place — no navigation, the editor stays mounted.
 */
export const OpenProjectDialog = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const projectIds = useWorkbenchSelector(
    (snapshot) => snapshot.state.projects.map((project) => project.id),
    areArraysEqual
  );
  const dispatch = useWorkbenchDispatch();
  const notify = useNotify();
  const summaries = useProjectLibrarySelector((snapshot) => snapshot.summaries);
  const [busyProjectId, setBusyProjectId] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      void refreshProjectLibrary();
    }
  }, [isOpen]);

  const openProjectIds = useMemo(() => new Set(projectIds), [projectIds]);
  const available = useMemo(
    () => summaries.filter((summary) => !openProjectIds.has(summary.id)),
    [openProjectIds, summaries]
  );

  const openProject = useCallback(
    async (summary: ProjectSummary) => {
      setBusyProjectId(summary.id);

      const project = await hydrateProjectFromServer(summary.id);

      setBusyProjectId(null);

      if (!project) {
        notify.error('Could not open project', `"${summary.name}" could not be loaded from the server.`);
        void refreshProjectLibrary();

        return;
      }

      flushGenerateDrafts();
      dispatch({ project, type: 'openProject' });
      onClose();
    },
    [dispatch, notify, onClose]
  );

  const handleImport = useCallback(async () => {
    const file = await pickProjectFile();

    if (!file) {
      return;
    }

    try {
      const record = await importProjectFile(file);
      const project = adoptProjectRecord(record);

      if (project) {
        flushGenerateDrafts();
        dispatch({ project, type: 'openProject' });
        onClose();
      }
    } catch (error) {
      notify.error('Import failed', error instanceof Error ? error.message : undefined);
    }
  }, [dispatch, notify, onClose]);

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  const startImport = useCallback(() => void handleImport(), [handleImport]);

  return (
    <Dialog.Root lazyMount open={isOpen} placement="center" size="sm" unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title fontSize="sm" fontWeight="700">
                Open project
              </Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Scrollable maxH="72">
                <Stack gap="1">
                  {available.map((summary) => (
                    <OpenProjectRow
                      key={summary.id}
                      isBusy={busyProjectId === summary.id}
                      isDisabled={busyProjectId !== null}
                      summary={summary}
                      onOpen={openProject}
                    />
                  ))}
                  {available.length === 0 ? (
                    <Text color="fg.muted" fontSize="xs" px="2.5" py="4" textAlign="center">
                      {summaries.length === 0 ? 'No saved projects yet.' : 'All saved projects are already open.'}
                    </Text>
                  ) : null}
                </Stack>
              </Scrollable>
            </Dialog.Body>
            <Dialog.Footer gap="2" justifyContent="space-between">
              <Button size="xs" variant="outline" onClick={startImport}>
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

const OpenProjectRow = ({
  isBusy,
  isDisabled,
  onOpen,
  summary,
}: {
  isBusy: boolean;
  isDisabled: boolean;
  onOpen: (summary: ProjectSummary) => Promise<void>;
  summary: ProjectSummary;
}) => {
  const open = useCallback(() => void onOpen(summary), [onOpen, summary]);

  return (
    <Row asChild gap="2.5" px="2.5" py="2" rounded="md" _disabled={disabledRowStyles}>
      <button disabled={isDisabled} type="button" onClick={open}>
        <Stack flex="1" gap="0" minW="0">
          <Text fontSize="xs" fontWeight="600" truncate>
            {summary.name}
          </Text>
          <Text color="fg.muted" fontSize="2xs">
            Edited {formatRelativeTime(summary.updatedAt)}
          </Text>
        </Stack>
        {isBusy ? <Spinner color="fg.muted" size="xs" /> : <Icon as={ArrowRightIcon} boxSize="3.5" color="fg.muted" />}
      </button>
    </Row>
  );
};
